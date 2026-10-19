# cython: language_level=3, boundscheck=False, wraparound=False
"""Low-level wrapper of the libopensps C API: one :class:`Context` per FSPS context handle.

Arrays come back as NumPy arrays whose memory the C library writes directly (no copies on the
way out). The C API is column-major; a C-ordered NumPy array of shape ``(n, nspec)`` has the
memory layout of the Fortran array ``(nspec, n)``, so every 2-D output here is indexed
``[model or age, pixel / band / line]``, as in python-fsps.
"""

import threading

import numpy as np

cimport numpy as cnp
from libc.stdlib cimport malloc, free

from . cimport _capi
from . import _config

cnp.import_array()

__all__ = ["Context", "FSPSError", "UnknownParameterError", "api_version", "header_api_version"]


# ----------------------------------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------------------------------

#: Meaning of the C status codes (include/fsps.h)
STATUS_MESSAGES = {
    1: "invalid context handle",
    2: ("unsupported: a parameter that changes the SSPs varied per model, tage <= 0 without "
        "SSP weights, or another option the batch does not support"),
    3: "unknown parameter or bad shape",
    101: "unknown integer parameter",
    102: "unknown float parameter",
    103: "unknown string parameter",
    301: "context not set up",
    302: "no CSP computed yet",
    304: "zmet out of range",
    306: "zcontinuous must be 0, 1, 2 or 3",
    309: "magnitudes or indices were not computed (fast mode)",
    311: "CSP computation failed",
}


class FSPSError(RuntimeError):
    """An error status from the C library; ``.status`` holds the code."""

    def __init__(self, status, message):
        super().__init__(f"{message} (status {status})")
        self.status = status


class UnknownParameterError(FSPSError, KeyError):
    """A parameter name the library does not know."""

    def __str__(self):
        return RuntimeError.__str__(self)


cdef str _last_error():
    cdef int st = 0
    cdef char buf[257]
    buf[0] = 0
    _capi.get_last_error(&st, buf, 256)
    return buf.decode("utf-8", "replace").strip()


cdef int _check(int status, str what) except -1:
    if status == 0:
        return 0
    msg = STATUS_MESSAGES.get(status, "error")
    detail = _last_error()
    _capi.clear_error()
    text = f"{what}: {msg}"
    if detail and detail not in text:
        text += f" [{detail}]"
    if status in (101, 102, 103):
        raise UnknownParameterError(status, text)
    raise FSPSError(status, text)


def api_version():
    """(major, minor, patch) of the loaded library's C API."""
    cdef int a = 0, b = 0, c = 0
    _capi.get_driver_version(&a, &b, &c)
    return (a, b, c)


def header_api_version():
    """(major, minor, patch) of the header these bindings were compiled against."""
    return (_capi.API_VERSION_MAJOR, _capi.API_VERSION_MINOR, _capi.API_VERSION_PATCH)


# ----------------------------------------------------------------------------------------------
# Locking. The library keeps its contexts in a pool that is reallocated when a context is
# created; no computation may run (with the GIL released) while that happens. Computations
# take the pool lock shared, creation and destruction exclusively. Each context also has its
# own lock: one context is not safe to use from two threads at once.
# ----------------------------------------------------------------------------------------------

class _PoolLock:
    def __init__(self):
        self._cond = threading.Condition()
        self._readers = 0

    def acquire_shared(self):
        with self._cond:
            self._readers += 1

    def release_shared(self):
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_exclusive(self):
        self._cond.acquire()
        while self._readers > 0:
            self._cond.wait()

    def release_exclusive(self):
        self._cond.release()


_pool_lock = _PoolLock()

# Parameters the C API takes as integers (fsps_context_set_int); everything else numeric is a
# float. 'sfh' is accepted as a float by the batch entry point.
INT_PARAMS = frozenset("""
sfh zmet wgp1 wgp2 wgp3 evtype imf_type tpagb_norm_type pzcon interpolation_type
add_agb_dust_model dust_type add_dust_emission compute_vega_mags vactoair_flag add_agn_dust
use_wr_spectra add_neb_emission add_neb_continuum cloudy_dust add_igm_absorption nebemlineinspec
add_xrb_emission add_stellar_remnants smooth_velocity smooth_lsf smoothspec_fast redshift_colors
compute_light_ages use_isoc_mdot setup_nebular_gaussians compute_mags compute_indices fast_mode
""".split())
STR_PARAMS = frozenset(["imf_filename", "sfh_filename"])

#: Outputs of a batch call and their per-model size attribute (None: one value)
BATCH_OUTPUTS = ("mags", "spec", "mass", "lbol", "mdust", "sfr", "emlines", "indx")


cdef inline double* _dptr(object a):
    if a is None:
        return NULL
    return <double*> cnp.PyArray_DATA(<cnp.ndarray> a)


def _as_scalar(value):
    """A scalar from a Python number or a size-1 array (as python-fsps accepts)."""
    a = np.asarray(value)
    if a.size != 1:
        raise ValueError(f"expected a scalar, got an array of shape {a.shape}")
    return a.reshape(()).item()


cdef class Context:
    """One FSPS context (``fsps_context_*`` handle): parameters, SSP grids and outputs.

    :param isoc, spec, dust: isochrone, spectral and dust-emission library names (e.g.
        ``"mist"``, ``"miles"``, ``"DL07"``); ``None`` for the library defaults.
    :param zin: metallicity index to load (1-based) or -1 for all (needed for ``logzsol``).
    :param params: parameters set before the setup (e.g. ``vactoair_flag``), then again after.
    :param setup: load the data now (default); otherwise call :meth:`setup` later.
    :param data_home: data root for this setup (overrides the configured one, see
        :func:`opensps.set_data_home`).
    """

    cdef readonly int handle
    cdef readonly bint is_setup
    cdef object _lock
    cdef dict _dims
    cdef object _wave, _ages, _zlegend, _emwave
    cdef readonly int last_batch_n

    def __cinit__(self, *args, **kwargs):
        cdef int h = 0, st = 0
        self.handle = 0
        self.is_setup = False
        self.last_batch_n = 0
        _pool_lock.acquire_exclusive()
        try:
            _capi.ctx_create(&h, &st)
        finally:
            _pool_lock.release_exclusive()
        _check(st, "fsps_context_create")
        self.handle = h
        self._lock = threading.RLock()
        self._dims = {}

    def __init__(self, isoc=None, spec=None, dust=None, zin=-1, params=None, setup=True,
                 data_home=None):
        if params:
            self.set_params(params)
        if setup:
            self.setup(isoc, spec, dust, zin=zin, data_home=data_home)
            if params:
                self.set_params(params)

    def __dealloc__(self):
        cdef int st = 0
        if self.handle > 0:
            _pool_lock.acquire_exclusive()
            try:
                _capi.ctx_destroy(self.handle, &st)
            finally:
                _pool_lock.release_exclusive()
            self.handle = 0

    def close(self):
        """Release the context's memory now (the object is unusable afterwards)."""
        cdef int st = 0
        if self.handle > 0:
            with self._lock:
                _pool_lock.acquire_exclusive()
                try:
                    _capi.ctx_destroy(self.handle, &st)
                finally:
                    _pool_lock.release_exclusive()
                self.handle = 0
                self.is_setup = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    cdef int _live(self) except -1:
        if self.handle <= 0:
            raise FSPSError(1, "context is closed")
        return 0

    cdef int _ready(self) except -1:
        self._live()
        if not self.is_setup:
            raise FSPSError(301, "context not set up: call setup() first")
        return 0

    # ------------------------------------------------------------------------------------------
    # Setup and parameters
    # ------------------------------------------------------------------------------------------

    def setup(self, isoc=None, spec=None, dust=None, zin=-1, data_home=None):
        """Load the libraries (expensive the first time per process; cached afterwards)."""
        cdef int st = 0
        cdef int z = zin
        self._live()
        bi = (isoc or "").encode()
        bs = (spec or "").encode()
        bd = (dust or "").encode()
        cdef const char *ci = bi
        cdef const char *cs = bs
        cdef const char *cd = bd
        cdef int h = self.handle
        with self._lock, _config.library_environment(data_home):
            _pool_lock.acquire_shared()
            try:
                with nogil:
                    _capi.ctx_setup(z, ci, cs, cd, h, &st)
            finally:
                _pool_lock.release_shared()
        _check(st, "fsps_context_setup")
        self.is_setup = True
        self._dims = {}
        self._wave = self._ages = self._zlegend = self._emwave = None

    def set_int(self, str name, value):
        cdef int st = 0
        self._live()
        b = name.encode()
        _capi.ctx_set_int(self.handle, b, int(value), &st)
        _check(st, f"set {name!r}")
        if name == "vactoair_flag":
            self._wave = self._emwave = None

    def set_float(self, str name, value):
        cdef int st = 0
        self._live()
        b = name.encode()
        _capi.ctx_set_float(self.handle, b, float(value), &st)
        _check(st, f"set {name!r}")

    def set_str(self, str name, str value):
        cdef int st = 0
        self._live()
        b = name.encode()
        v = value.encode()
        _capi.ctx_set_str(self.handle, b, v, &st)
        _check(st, f"set {name!r}")

    def set(self, str name, value):
        """Set one parameter by name; the type (int, float, string) follows the name.

        Booleans are passed as 0 / 1. Size-1 arrays are accepted as scalars.
        """
        if name in STR_PARAMS:
            self.set_str(name, str(value))
            return
        v = _as_scalar(value)
        if name in INT_PARAMS:
            if isinstance(v, float) and not v.is_integer():
                raise ValueError(f"{name!r} takes an integer, got {v!r}")
            self.set_int(name, int(v))
        else:
            self.set_float(name, float(v))

    def set_params(self, params=None, **kwargs):
        """Set several parameters from a mapping and / or keyword arguments."""
        if params:
            for k, v in dict(params).items():
                self.set(k, v)
        for k, v in kwargs.items():
            self.set(k, v)

    def set_fast_mode(self, bint on):
        """Fast mode: no magnitudes or indices in single CSPs (fewer post-processing steps)."""
        cdef int st = 0
        self._live()
        _capi.ctx_set_fast_mode(self.handle, 1 if on else 0, &st)
        _check(st, "set_fast_mode")

    def set_tabular_sfh(self, age, sfr, met=None):
        """Tabular SFH for ``sfh = 3``: ``age`` [Gyr since the start], ``sfr`` [M_sun/yr],
        optionally ``met`` (Z, used with ``zcontinuous = 3``)."""
        cdef int st = 0
        self._live()
        a = np.ascontiguousarray(age, dtype=np.float64).ravel()
        s = np.ascontiguousarray(sfr, dtype=np.float64).ravel()
        m = None if met is None else np.ascontiguousarray(met, dtype=np.float64).ravel()
        if a.shape != s.shape or (m is not None and m.shape != a.shape):
            raise ValueError("age, sfr (and met) must have the same length")
        _capi.ctx_set_sfh_tab(self.handle, a.size, _dptr(a), _dptr(s), _dptr(m), &st)
        _check(st, "set_tabular_sfh")

    # ------------------------------------------------------------------------------------------
    # Dimensions and grids
    # ------------------------------------------------------------------------------------------

    cdef int _dim(self, str name) except -1:
        cdef int n = 0, st = 0
        self._ready()
        if name in self._dims:
            return self._dims[name]
        if name == "nspec":
            _capi.ctx_get_nspec(self.handle, &n, &st)
        elif name == "ntfull":
            _capi.ctx_get_ntfull(self.handle, &n, &st)
        elif name == "nbands":
            _capi.ctx_get_nbands(self.handle, &n, &st)
        elif name == "nindx":
            _capi.ctx_get_nindx(self.handle, &n, &st)
        elif name == "nz":
            _capi.ctx_get_nz(self.handle, &n, &st)
        elif name == "nemline":
            _capi.ctx_get_nemline(self.handle, &n, &st)
        _check(st, f"get {name}")
        self._dims[name] = n
        return n

    @property
    def nspec(self):
        return self._dim("nspec")

    @property
    def ntfull(self):
        return self._dim("ntfull")

    @property
    def nbands(self):
        return self._dim("nbands")

    @property
    def nindx(self):
        return self._dim("nindx")

    @property
    def nz(self):
        return self._dim("nz")

    @property
    def nemline(self):
        return self._dim("nemline")

    @property
    def zsol(self):
        """Solar metallicity (mass fraction) of the isochrone set."""
        cdef double z = 0.0
        cdef int st = 0
        self._ready()
        _capi.ctx_get_zsol(self.handle, &z, &st)
        _check(st, "get zsol")
        return z

    @property
    def paths(self):
        """(sps_home, data_home, output_home) as resolved by the library at setup."""
        cdef char a[1025]
        cdef char b[1025]
        cdef char c[1025]
        self._live()
        a[0] = b[0] = c[0] = 0
        _capi.ctx_get_paths(self.handle, a, 1024, b, 1024, c, 1024)
        return (a.decode(), b.decode(), c.decode())

    def _grid(self, str which):
        cdef int st = 0
        self._ready()
        if which == "lambda":
            out = np.empty(self.nspec)
            _capi.ctx_get_lambda(self.handle, _dptr(out), &st)
        elif which == "ages":
            out = np.empty(self.ntfull)
            _capi.ctx_get_ages(self.handle, _dptr(out), &st)
        elif which == "zlegend":
            out = np.empty(self.nz)
            _capi.ctx_get_zlegend(self.handle, _dptr(out), &st)
        else:
            out = np.empty(self.nemline)
            _capi.ctx_get_emlambda(self.handle, _dptr(out), &st)
        _check(st, f"get {which}")
        out.flags.writeable = False
        return out

    @property
    def wavelengths(self):
        """Wavelengths of the spectra [A] (air with ``vactoair_flag = 1``), read-only."""
        if self._wave is None:
            self._wave = self._grid("lambda")
        return self._wave

    @property
    def log_ages(self):
        """SSP ages, log10(age / yr), read-only."""
        if self._ages is None:
            self._ages = self._grid("ages")
        return self._ages

    @property
    def zlegend(self):
        """Metallicities (Z) of the isochrone grid, read-only."""
        if self._zlegend is None:
            self._zlegend = self._grid("zlegend")
        return self._zlegend

    @property
    def emline_wavelengths(self):
        """Emission-line wavelengths [A], read-only."""
        if self._emwave is None:
            self._emwave = self._grid("emlambda")
        return self._emwave

    def filter_data(self):
        """(effective wavelength [A], AB magnitude of Vega, absolute AB magnitude of the Sun)."""
        cdef int st = 0
        n = self.nbands
        a, b, c = np.empty(n), np.empty(n), np.empty(n)
        _capi.ctx_get_filter_data(self.handle, _dptr(a), _dptr(b), _dptr(c), &st)
        _check(st, "get filter data")
        return a, b, c

    # ------------------------------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------------------------------

    def compute_ssp(self):
        """The SSP grid at the current parameters (``zmet``).

        :returns: ``spec`` (ntfull, nspec) [L_sun/Hz per M_sun formed], ``mass`` (ntfull)
            [M_sun], ``lbol`` (ntfull) [log10 L_sun].
        """
        cdef int st = 0
        cdef int h = self.handle
        self._ready()
        spec = np.empty((self.ntfull, self.nspec))
        mass = np.empty(self.ntfull)
        lbol = np.empty(self.ntfull)
        cdef double *ps = _dptr(spec)
        cdef double *pm = _dptr(mass)
        cdef double *pl = _dptr(lbol)
        with self._lock:
            _pool_lock.acquire_shared()
            try:
                with nogil:
                    _capi.ctx_compute_ssp(h, ps, pm, pl, &st)
            finally:
                _pool_lock.release_shared()
        _check(st, "compute_ssp")
        return spec, mass, lbol

    def compute_csp(self, int zcontinuous=0, outputs=None):
        """One CSP at the current parameters (the reference single-model path).

        :param zcontinuous: 0 SSPs at ``zmet``; 1 interpolated at ``logzsol``; 2 a metallicity
            distribution (``pmetals``); 3 all metallicities (tabular SFH metallicity).
        :param outputs: names to return (default: all available) among ``age, mass, lbol, sfr,
            mdust, mformed, spec, mags, emlines, indx``.
        :returns: dict of arrays with a leading axis of length ``n_out`` (1 for ``tage > 0``,
            else ``ntfull``).
        """
        cdef int st = 0, n_out = 0
        cdef int h = self.handle
        cdef int zc = zcontinuous
        self._ready()
        with self._lock:
            _pool_lock.acquire_shared()
            try:
                with nogil:
                    _capi.ctx_compute_csp(h, zc, &n_out, &st)
            finally:
                _pool_lock.release_shared()
            _check(st, "compute_csp")
            return self._get_csp(n_out, outputs)

    def _get_csp(self, int n, outputs=None):
        cdef int st = 0
        names = ("age", "mass", "lbol", "sfr", "mdust", "mformed", "spec", "mags", "emlines", "indx")
        want = set(names if outputs is None else outputs)
        unknown = want - set(names)
        if unknown:
            raise ValueError(f"unknown outputs {sorted(unknown)}")
        sizes = {"spec": self.nspec, "mags": self.nbands, "emlines": self.nemline,
                 "indx": self.nindx}
        out = {}
        for k in names:
            if k in want:
                out[k] = np.empty((n, sizes[k])) if k in sizes else np.empty(n)
            else:
                out[k] = None
        _capi.ctx_get_csp(self.handle, _dptr(out["age"]), _dptr(out["mass"]), _dptr(out["lbol"]),
                          _dptr(out["sfr"]), _dptr(out["mdust"]), _dptr(out["mformed"]),
                          _dptr(out["spec"]), _dptr(out["mags"]), _dptr(out["emlines"]),
                          _dptr(out["indx"]), &st)
        if st == 309 and outputs is None:
            # Fast mode: no magnitudes or indices; return the rest
            out["mags"] = out["indx"] = None
            _capi.clear_error()
            _capi.ctx_get_csp(self.handle, _dptr(out["age"]), _dptr(out["mass"]),
                              _dptr(out["lbol"]), _dptr(out["sfr"]), _dptr(out["mdust"]),
                              _dptr(out["mformed"]), _dptr(out["spec"]), NULL,
                              _dptr(out["emlines"]), NULL, &st)
        _check(st, "get_csp")
        return {k: v for k, v in out.items() if v is not None}

    def compute_csp_batch(self, params=None, ssp_weights=None, n=None, spec=True,
                          emlines=False, indices=False, out=None):
        """Many CSPs in one call (``fsps_context_compute_csp_batch``).

        The models share the context's parameters except those in ``params``.

        :param params: mapping of parameter name to a 1-D array of ``n`` values (scalars are
            broadcast). Only per-model parameters (SFH, dust, dust emission, nebular gas, AGN,
            ``logzsol``, ``zred``, ``igm_factor``, ...) may vary; parameters that change the SSPs
            (IMF, isochrone options) raise :class:`FSPSError` (status 2). ``sfh`` may be given
            as numbers.
        :param ssp_weights: optional (n, ntfull) array of SSP weights (M_sun formed per SSP
            age), replacing the SFH parameters (non-parametric SFHs).
        :param n: number of models, if it cannot be inferred from the arrays.
        :param spec, emlines, indices: also return spectra / line luminosities / indices
            (magnitudes, mass, lbol, mdust and sfr always come back).
        :param out: optional dict of preallocated C-contiguous float64 arrays (same shapes as
            the result) to write into, for repeated calls without allocation.
        :returns: dict with ``mags`` (n, nbands), ``mass``, ``lbol``, ``mdust``, ``sfr`` (n), and
            optionally ``spec`` (n, nspec), ``emlines`` (n, nemline), ``indx`` (n, nindx).
        """
        cdef int st = 0
        cdef int i, npar, nn, h
        cdef const char **cnames = NULL
        cdef const double **cvalues = NULL
        cdef double *pw = NULL
        self._ready()
        params = dict(params or {})
        lengths = {np.size(v) for v in params.values() if np.ndim(v) > 0}
        if ssp_weights is not None:
            ssp_weights = np.ascontiguousarray(ssp_weights, dtype=np.float64)
            if ssp_weights.ndim == 1:
                ssp_weights = ssp_weights[None, :]
            if ssp_weights.ndim != 2 or ssp_weights.shape[1] != self.ntfull:
                raise ValueError(f"ssp_weights must have shape (n, {self.ntfull})")
            lengths.add(ssp_weights.shape[0])
        if n is not None:
            lengths.add(int(n))
        if len(lengths) > 1:
            raise ValueError(f"inconsistent numbers of models: {sorted(lengths)}")
        if not lengths:
            raise ValueError("give n, an array parameter or ssp_weights")
        nn = lengths.pop()
        if nn <= 0:
            raise ValueError("n must be positive")

        keep = []        # keeps the encoded names and value arrays alive during the call
        names_b = []
        arrays = []
        for k, v in params.items():
            a = np.ascontiguousarray(np.broadcast_to(np.asarray(v, dtype=np.float64), (nn,)))
            names_b.append(k.encode())
            arrays.append(a)
        npar = len(arrays)

        shapes = {"mags": (nn, self.nbands), "mass": (nn,), "lbol": (nn,), "mdust": (nn,),
                  "sfr": (nn,)}
        if spec:
            shapes["spec"] = (nn, self.nspec)
        if emlines:
            shapes["emlines"] = (nn, self.nemline)
        if indices:
            shapes["indx"] = (nn, self.nindx)
        res = {}
        out = out or {}
        for k, shp in shapes.items():
            a = out.get(k)
            if a is None:
                a = np.empty(shp)
            elif (not isinstance(a, np.ndarray) or a.shape != shp or a.dtype != np.float64
                  or not a.flags.c_contiguous or not a.flags.writeable):
                raise ValueError(f"out[{k!r}] must be a writeable C-contiguous float64 array of "
                                 f"shape {shp}")
            res[k] = a

        cdef double *pmags = _dptr(res["mags"])
        cdef double *pspec = _dptr(res.get("spec"))
        cdef double *pmass = _dptr(res["mass"])
        cdef double *plbol = _dptr(res["lbol"])
        cdef double *pmdust = _dptr(res["mdust"])
        cdef double *psfr = _dptr(res["sfr"])
        cdef double *peml = _dptr(res.get("emlines"))
        cdef double *pindx = _dptr(res.get("indx"))
        pw = _dptr(ssp_weights)
        h = self.handle
        if npar > 0:
            cnames = <const char **> malloc(npar * sizeof(char *))
            cvalues = <const double **> malloc(npar * sizeof(double *))
            if cnames == NULL or cvalues == NULL:
                free(cnames)
                free(cvalues)
                raise MemoryError()
            for i in range(npar):
                cnames[i] = <const char *> names_b[i]
                cvalues[i] = _dptr(arrays[i])
        try:
            with self._lock:
                _pool_lock.acquire_shared()
                try:
                    with nogil:
                        _capi.ctx_compute_csp_batch(h, nn, npar, cnames, cvalues, pw, pmags,
                                                    pspec, pmass, plbol, pmdust, psfr, peml,
                                                    pindx, &st)
                finally:
                    _pool_lock.release_shared()
        finally:
            free(cnames)
            free(cvalues)
        _check(st, "compute_csp_batch")
        self.last_batch_n = nn
        return res

    def batch_spectra_host(self, copy=False):
        """The last batch's spectra in kernel precision (float32 in FP32-kernel builds),
        shape (n, nspec), without conversion to double.

        With ``copy=False`` the array is a read-only view of a buffer owned by the context,
        valid only until the next batch call on this context.
        """
        cdef void *p = NULL
        cdef int ns = 0, nm = 0, nb = 0, st = 0
        cdef cnp.npy_intp dims[2]
        self._ready()
        _capi.ctx_batch_spec_host(self.handle, &p, &ns, &nm, &nb, &st)
        _check(st, "batch_spec_host")
        if p == NULL or nm == 0:
            raise FSPSError(302, "no batch spectra (run compute_csp_batch first)")
        dims[0] = nm
        dims[1] = ns
        typenum = cnp.NPY_FLOAT32 if nb == 4 else cnp.NPY_FLOAT64
        arr = cnp.PyArray_SimpleNewFromData(2, dims, typenum, p)
        cnp.PyArray_SetBaseObject(<cnp.ndarray> arr, self)
        cnp.Py_INCREF(self)
        arr.flags.writeable = False
        return arr.copy() if copy else arr

    def batch_spectra_device(self):
        """Where the last batch's spectra live (on the GPU in GPU builds): a
        :class:`DeviceSpectra` with ``ptr``, ``shape`` (n, nspec), ``dtype`` and
        ``__cuda_array_interface__`` for CuPy, Numba or PyTorch. In CPU builds the address is
        a host address; use :meth:`batch_spectra_host` there. Valid until the next batch."""
        cdef void *p = NULL
        cdef int ns = 0, nm = 0, nb = 0, st = 0
        self._ready()
        _capi.ctx_batch_spec_device(self.handle, &p, &ns, &nm, &nb, &st)
        _check(st, "batch_spec_device")
        if p == NULL or nm == 0:
            raise FSPSError(302, "no batch spectra (run compute_csp_batch first)")
        return DeviceSpectra(<size_t> p, (nm, ns), np.dtype(np.float32 if nb == 4 else np.float64),
                             self)


class DeviceSpectra:
    """Address, shape and type of the batch spectra on the device (see
    :meth:`Context.batch_spectra_device`). Holds a reference to its context."""

    def __init__(self, ptr, shape, dtype, owner):
        self.ptr = ptr
        self.shape = shape
        self.dtype = dtype
        self._owner = owner

    @property
    def __cuda_array_interface__(self):
        return {"shape": self.shape, "typestr": self.dtype.str, "data": (self.ptr, True),
                "version": 3, "strides": None}

    def __repr__(self):
        return f"DeviceSpectra(ptr=0x{self.ptr:x}, shape={self.shape}, dtype={self.dtype})"
