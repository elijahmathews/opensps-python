"""The C API called directly through ctypes, independently of the Cython extension: the
reference the bindings are compared with (bit for bit).

The library is the one the extension uses: the extension itself when the core is linked in
statically (subproject build; it exports the fsps_* symbols), else libfsps / libopensps found
by the loader. OPENSPS_TEST_CLIB overrides the path.
"""

import ctypes
import ctypes.util
import os

import numpy as np

import opensps._config as _config
import opensps._core as _core

D = ctypes.POINTER(ctypes.c_double)
I = ctypes.POINTER(ctypes.c_int)


def _load():
    path = os.environ.get("OPENSPS_TEST_CLIB")
    if path:
        return ctypes.CDLL(path)
    lib = ctypes.CDLL(_core.__file__)
    if hasattr(lib, "fsps_context_create"):
        return lib
    for name in ("opensps", "fsps"):
        found = ctypes.util.find_library(name)
        if found:
            return ctypes.CDLL(found)
    raise RuntimeError("core library not found; set OPENSPS_TEST_CLIB")


def _p(a):
    return None if a is None else a.ctypes.data_as(D)


class CRef:
    def __init__(self):
        self.lib = _load()

    def call(self, name, *args):
        st = ctypes.c_int(0)
        getattr(self.lib, name)(*args, ctypes.byref(st))
        assert st.value == 0, f"{name}: status {st.value}"

    def create(self, isoc="mist", spec="miles", dust="DL07"):
        h = ctypes.c_int(0)
        self.call("fsps_context_create", ctypes.byref(h))
        with _config.library_environment():  # the data root the bindings use
            self.call("fsps_context_setup", ctypes.c_int(-1), isoc.encode(), spec.encode(),
                      dust.encode(), h)
        return h.value

    def destroy(self, h):
        self.call("fsps_context_destroy", ctypes.c_int(h))

    def set(self, h, name, value, as_int):
        if as_int:
            self.call("fsps_context_set_int", ctypes.c_int(h), name.encode(), ctypes.c_int(int(value)))
        else:
            self.call("fsps_context_set_float", ctypes.c_int(h), name.encode(),
                      ctypes.c_double(float(value)))

    def dim(self, h, what):
        n = ctypes.c_int(0)
        self.call(f"fsps_context_get_{what}", ctypes.c_int(h), ctypes.byref(n))
        return n.value

    def grid(self, h, what, n):
        out = np.empty(n)
        self.call(f"fsps_context_get_{what}", ctypes.c_int(h), _p(out))
        return out

    def compute_ssp(self, h):
        ns, nt = self.dim(h, "nspec"), self.dim(h, "ntfull")
        spec, mass, lbol = np.empty((nt, ns)), np.empty(nt), np.empty(nt)
        self.call("fsps_context_compute_ssp", ctypes.c_int(h), _p(spec), _p(mass), _p(lbol))
        return spec, mass, lbol

    def compute_csp(self, h, zc):
        n = ctypes.c_int(0)
        self.call("fsps_context_compute_csp", ctypes.c_int(h), ctypes.c_int(zc), ctypes.byref(n))
        n = n.value
        ns, nb, nl, ni = (self.dim(h, k) for k in ("nspec", "nbands", "nemline", "nindx"))
        out = {k: np.empty(n) for k in ("age", "mass", "lbol", "sfr", "mdust", "mformed")}
        out.update(spec=np.empty((n, ns)), mags=np.empty((n, nb)), emlines=np.empty((n, nl)),
                   indx=np.empty((n, ni)))
        self.call("fsps_context_get_csp", ctypes.c_int(h), *[_p(out[k]) for k in (
            "age", "mass", "lbol", "sfr", "mdust", "mformed", "spec", "mags", "emlines", "indx")])
        return out

    def batch(self, h, n, params, weights=None):
        ns, nb, nl = (self.dim(h, k) for k in ("nspec", "nbands", "nemline"))
        names = (ctypes.c_char_p * len(params))(*[k.encode() for k in params])
        arrays = [np.ascontiguousarray(np.broadcast_to(np.asarray(v, float), (n,))) for v in params.values()]
        values = (D * len(params))(*[_p(a) for a in arrays])
        out = dict(mags=np.empty((n, nb)), spec=np.empty((n, ns)), mass=np.empty(n),
                   lbol=np.empty(n), mdust=np.empty(n), sfr=np.empty(n), emlines=np.empty((n, nl)))
        w = None if weights is None else np.ascontiguousarray(weights, float)
        self.call("fsps_context_compute_csp_batch", ctypes.c_int(h), ctypes.c_int(n),
                  ctypes.c_int(len(params)), names, values, _p(w), _p(out["mags"]), _p(out["spec"]),
                  _p(out["mass"]), _p(out["lbol"]), _p(out["mdust"]), _p(out["sfr"]),
                  _p(out["emlines"]), None)
        return out
