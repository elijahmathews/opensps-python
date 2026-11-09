"""OpenSPS Context object managing state, grids, and calculations."""

import numpy as np
from ctypes import c_int, c_double, c_char_p, byref
from typing import Dict, Optional, Sequence, Union, Tuple, Any
from opensps import _c_api
from opensps.batch import CSPBatch, CSPBatchResult
from opensps.ssp import SSPOperator
from opensps.constants import OpenSPSError, FeatureNotSupportedError


class Context:
    """An isolated, thread-safe OpenSPS calculation context.

    Parameters
    ----------
    zmet : int, optional
        Metallicity index (1 to nz). Pass -1 to load all metallicities for interpolation.
    isoc : str, optional
        Isochrone library identifier (e.g. "mist", "pdva", "parsec"). Default: "mist".
    spec : str, optional
        Spectral library identifier (e.g. "miles", "basel", "c3k"). Default: "miles".
    dust : str, optional
        Dust emission library identifier (e.g. "DL07", "themis"). Default: "DL07".
    auto_setup : bool, optional
        Whether to call setup immediately. Default: True.
    """

    def __init__(
        self,
        zmet: int = -1,
        isoc: str = "mist",
        spec: str = "miles",
        dust: str = "DL07",
        auto_setup: bool = True,
    ):
        self._handle = None
        self._ssp_op = None
        self._cached_grids = {}

        handle = c_int(0)
        status = c_int(0)
        _c_api._LIB.opensps_context_create(byref(handle), byref(status))
        _c_api.check_status(status.value, "context_create")
        self._handle = handle.value

        if auto_setup:
            self.setup(zmet, isoc, spec, dust)

    @property
    def handle(self) -> int:
        if self._handle is None:
            raise RuntimeError("Context has been destroyed.")
        return self._handle

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()

    def __del__(self):
        self.destroy()

    def destroy(self):
        """Release context memory in libopensps."""
        if self._handle is not None:
            if self._ssp_op is not None:
                try:
                    self._ssp_op.free()
                except Exception:
                    pass
                self._ssp_op = None

            status = c_int(0)
            _c_api._LIB.opensps_context_destroy(self._handle, byref(status))
            self._handle = None

    def setup(self, zmet: int = -1, isoc: str = "mist", spec: str = "miles", dust: str = "DL07"):
        """Load library grids and initialize physics for this context."""
        status = c_int(0)
        _c_api._LIB.opensps_context_setup(
            zmet,
            isoc.encode("utf-8"),
            spec.encode("utf-8"),
            dust.encode("utf-8"),
            self.handle,
            byref(status),
        )
        _c_api.check_status(status.value, f"context_setup({zmet}, '{isoc}', '{spec}', '{dust}')")
        self._cached_grids.clear()

    # -------------------------------------------------------------------------
    # Dimensions
    # -------------------------------------------------------------------------
    @property
    def nspec(self) -> int:
        val = c_int(0)
        status = c_int(0)
        _c_api._LIB.opensps_context_get_nspec(self.handle, byref(val), byref(status))
        _c_api.check_status(status.value, "get_nspec")
        return val.value

    @property
    def ntfull(self) -> int:
        val = c_int(0)
        status = c_int(0)
        _c_api._LIB.opensps_context_get_ntfull(self.handle, byref(val), byref(status))
        _c_api.check_status(status.value, "get_ntfull")
        return val.value

    @property
    def nbands(self) -> int:
        val = c_int(0)
        status = c_int(0)
        _c_api._LIB.opensps_context_get_nbands(self.handle, byref(val), byref(status))
        _c_api.check_status(status.value, "get_nbands")
        return val.value

    @property
    def nindx(self) -> int:
        val = c_int(0)
        status = c_int(0)
        _c_api._LIB.opensps_context_get_nindx(self.handle, byref(val), byref(status))
        _c_api.check_status(status.value, "get_nindx")
        return val.value

    @property
    def nz(self) -> int:
        val = c_int(0)
        status = c_int(0)
        _c_api._LIB.opensps_context_get_nz(self.handle, byref(val), byref(status))
        _c_api.check_status(status.value, "get_nz")
        return val.value

    @property
    def nemline(self) -> int:
        val = c_int(0)
        status = c_int(0)
        _c_api._LIB.opensps_context_get_nemline(self.handle, byref(val), byref(status))
        _c_api.check_status(status.value, "get_nemline")
        return val.value

    @property
    def zsol(self) -> float:
        val = c_double(0.0)
        status = c_int(0)
        _c_api._LIB.opensps_context_get_zsol(self.handle, byref(val), byref(status))
        _c_api.check_status(status.value, "get_zsol")
        return val.value

    # -------------------------------------------------------------------------
    # Grids
    # -------------------------------------------------------------------------
    @property
    def wavelengths(self) -> np.ndarray:
        """Rest-frame wavelength grid in Angstroms (nspec)."""
        if "wavelengths" not in self._cached_grids:
            arr = np.empty(self.nspec, dtype=np.float64)
            status = c_int(0)
            _c_api._LIB.opensps_context_get_lambda(
                self.handle, arr.ctypes.data_as(_c_api.POINTER(c_double)), byref(status)
            )
            _c_api.check_status(status.value, "get_lambda")
            self._cached_grids["wavelengths"] = arr
        return self._cached_grids["wavelengths"]

    @property
    def log_ages(self) -> np.ndarray:
        """SSP log10(age/yr) grid (ntfull)."""
        if "log_ages" not in self._cached_grids:
            arr = np.empty(self.ntfull, dtype=np.float64)
            status = c_int(0)
            _c_api._LIB.opensps_context_get_ages(
                self.handle, arr.ctypes.data_as(_c_api.POINTER(c_double)), byref(status)
            )
            _c_api.check_status(status.value, "get_ages")
            self._cached_grids["log_ages"] = arr
        return self._cached_grids["log_ages"]

    @property
    def zlegend(self) -> np.ndarray:
        """Metallicity grid in mass fraction Z (nz)."""
        if "zlegend" not in self._cached_grids:
            arr = np.empty(self.nz, dtype=np.float64)
            status = c_int(0)
            _c_api._LIB.opensps_context_get_zlegend(
                self.handle, arr.ctypes.data_as(_c_api.POINTER(c_double)), byref(status)
            )
            _c_api.check_status(status.value, "get_zlegend")
            self._cached_grids["zlegend"] = arr
        return self._cached_grids["zlegend"]

    @property
    def emlambda(self) -> np.ndarray:
        """Emission line wavelengths in Angstroms (nemline)."""
        if "emlambda" not in self._cached_grids:
            arr = np.empty(self.nemline, dtype=np.float64)
            status = c_int(0)
            _c_api._LIB.opensps_context_get_emlambda(
                self.handle, arr.ctypes.data_as(_c_api.POINTER(c_double)), byref(status)
            )
            _c_api.check_status(status.value, "get_emlambda")
            self._cached_grids["emlambda"] = arr
        return self._cached_grids["emlambda"]

    # -------------------------------------------------------------------------
    # Parameter Setters
    # -------------------------------------------------------------------------
    def set_param(self, key: str, value: Union[int, float, str]):
        """Set a parameter on this context."""
        status = c_int(0)
        c_key = key.encode("utf-8")
        if isinstance(value, int):
            _c_api._LIB.opensps_context_set_int(self.handle, c_key, value, byref(status))
        elif isinstance(value, float):
            _c_api._LIB.opensps_context_set_float(self.handle, c_key, value, byref(status))
        elif isinstance(value, str):
            _c_api._LIB.opensps_context_set_str(self.handle, c_key, value.encode("utf-8"), byref(status))
        elif isinstance(value, bool):
            _c_api._LIB.opensps_context_set_int(self.handle, c_key, int(value), byref(status))
        else:
            # Fall back to float conversion
            _c_api._LIB.opensps_context_set_float(self.handle, c_key, float(value), byref(status))

        _c_api.check_status(status.value, f"set_param('{key}', {value})")

    def __setitem__(self, key: str, value: Union[int, float, str]):
        self.set_param(key, value)

    def set_mag_compute(self, bands: Optional[Sequence[int]] = None):
        """Choose the bands whose magnitudes the context computes (batches and scenarios).

        The photometry costs in proportion to the number of bands computed, so a fit that uses a few
        of the library's filters (159 in the default set) should select them. The magnitudes of the
        bands not selected are NaN.

        Parameters
        ----------
        bands : array_like of int, optional
            Zero-based indices of the bands to compute (as in the rows of the magnitudes, see
            :attr:`nbands`). ``None`` computes every band again.
        """
        if not _c_api.HAS_MAG_COMPUTE:
            raise RuntimeError("this libopensps does not provide opensps_context_set_mag_compute")
        status = c_int(0)
        if bands is None:
            _c_api._LIB.opensps_context_set_mag_compute(self.handle, 0, None, byref(status))
        else:
            idx = np.asarray(bands, dtype=np.int64).reshape(-1)
            nb = self.nbands
            if idx.size and (idx.min() < 0 or idx.max() >= nb):
                raise ValueError(f"band indices must be in [0, {nb - 1}]")
            mask = np.zeros(nb, dtype=np.int32)
            mask[idx] = 1
            _c_api._LIB.opensps_context_set_mag_compute(
                self.handle, nb, mask.ctypes.data_as(_c_api.POINTER(c_int)), byref(status)
            )
        _c_api.check_status(status.value, "set_mag_compute")

    def set_fast_mode(self, fast_mode: bool):
        """Enable or disable fast mode (skips magnitudes and spectral indices in CSP calls)."""
        status = c_int(0)
        _c_api._LIB.opensps_context_set_fast_mode(self.handle, 1 if fast_mode else 0, byref(status))
        _c_api.check_status(status.value, f"set_fast_mode({fast_mode})")

    # -------------------------------------------------------------------------
    # Batch Calculations & Operators
    # -------------------------------------------------------------------------
    def batch(
        self,
        params: Optional[Dict[str, Any]] = None,
        ssp_weights: Optional[np.ndarray] = None,
        ssp_weights_z: Optional[np.ndarray] = None,
        zred_mags: Optional[Union[Sequence[float], np.ndarray]] = None,
        **kwargs,
    ) -> CSPBatch:
        """Create a high-performance batch runner bound to this context."""
        return CSPBatch(
            self,
            params=params,
            ssp_weights=ssp_weights,
            ssp_weights_z=ssp_weights_z,
            zred_mags=zred_mags,
            **kwargs,
        )

    def build_ssp_operator(self, all_metallicities: bool = False) -> SSPOperator:
        """Build and enable an SSP operator on this context for rapid IMF variations."""
        if not _c_api.HAS_SSP_OPS:
            raise FeatureNotSupportedError(
                "SSP operators require a libopensps shared library built with SSP operator support."
            )
        self._ssp_op = SSPOperator(self, all_metallicities=all_metallicities)
        return self._ssp_op

    @property
    def has_ssp_operator(self) -> bool:
        """Whether this context currently has an active SSP operator."""
        if not _c_api.HAS_SSP_OPS:
            return False
        val = c_int(0)
        status = c_int(0)
        _c_api._LIB.opensps_context_has_ssp_ops(self.handle, byref(val), byref(status))
        _c_api.check_status(status.value, "has_ssp_ops")
        return val.value == 1

    def free_ssp_operator(self):
        """Release context-managed SSP operators."""
        if not _c_api.HAS_SSP_OPS:
            return
        if self._ssp_op is not None:
            self._ssp_op.free()
            self._ssp_op = None
        else:
            status = c_int(0)
            _c_api._LIB.opensps_context_free_ssp_ops(self.handle, byref(status))
            _c_api.check_status(status.value, "free_ssp_ops")

    def evaluate_ssp_operator(
        self,
        imf_type: Optional[int] = None,
        imf_params: Optional[Sequence[float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate active SSP operator for an IMF."""
        if not _c_api.HAS_SSP_OPS:
            raise FeatureNotSupportedError(
                "SSP operators require a libopensps shared library built with SSP operator support."
            )
        if self._ssp_op is None:
            if not self.has_ssp_operator:
                raise OpenSPSError(303, "No SSP operator has been built on this context.")
            self._ssp_op = SSPOperator.__new__(SSPOperator)
            self._ssp_op._context = self
            self._ssp_op.all_metallicities = False
        return self._ssp_op.evaluate(imf_type=imf_type, imf_params=imf_params)

    def get_batch_spec_device_ptr(self) -> Tuple[int, int, int, int]:
        """Get the on-device memory address of the last computed batch spectra.

        Returns
        -------
        device_ptr : int
            Raw GPU device pointer address (suitable for zero-copy wrapping via HIP/ROCm/PyTorch).
        nspec : int
            Number of spectral wavelengths.
        n_models : int
            Number of models in the batch.
        bytes_per_elem : int
            Bytes per value (4 for FP32, 8 for FP64).
        """
        if not _c_api.HAS_DEVICE_PTR:
            raise FeatureNotSupportedError(
                "Device pointer access requires a libopensps shared library built with GPU offload support."
            )
        ptr = _c_api.c_void_p()
        ns = _c_api.c_int(0)
        nm = _c_api.c_int(0)
        nb = _c_api.c_int(0)
        status = _c_api.c_int(0)
        _c_api._LIB.opensps_context_batch_spec_device(
            self.handle, _c_api.byref(ptr), _c_api.byref(ns), _c_api.byref(nm), _c_api.byref(nb), _c_api.byref(status)
        )
        _c_api.check_status(status.value, "batch_spec_device")
        addr = ptr.value if ptr.value is not None else 0
        return addr, ns.value, nm.value, nb.value

    # -------------------------------------------------------------------------
    # Single Computations
    # -------------------------------------------------------------------------
    def compute_ssp(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the SSP grid of the context's parameters.

        Returns
        -------
        spec : ndarray of shape (ntfull, nspec)
        mass : ndarray of shape (ntfull,)
        lbol : ndarray of shape (ntfull,)
        """
        ns = self.nspec
        nt = self.ntfull
        spec = np.empty((ns, nt), dtype=np.float64, order="F")
        mass = np.empty(nt, dtype=np.float64)
        lbol = np.empty(nt, dtype=np.float64)

        status = c_int(0)
        _c_api._LIB.opensps_context_compute_ssp(
            self.handle,
            spec.ctypes.data_as(_c_api.POINTER(c_double)),
            mass.ctypes.data_as(_c_api.POINTER(c_double)),
            lbol.ctypes.data_as(_c_api.POINTER(c_double)),
            byref(status),
        )
        _c_api.check_status(status.value, "compute_ssp")
        return spec.T.copy(), mass, lbol
