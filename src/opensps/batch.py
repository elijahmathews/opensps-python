"""High-throughput batched CSP calculation and reverse-mode automatic differentiation."""

from dataclasses import dataclass
import numpy as np
from ctypes import c_int, c_double, c_char_p, byref, POINTER
from typing import Dict, Optional, Sequence, Union, Tuple, Any
from opensps import _c_api
from opensps.constants import BGRAD, BGRAD_TO_PARAM, NBGRAD, OpenSPSError


@dataclass
class CSPBatchResult:
    """Outputs from a forward batched CSP evaluation."""
    mags: np.ndarray                      # (N, nbands)
    spec: Optional[np.ndarray] = None     # (N, nspec) if requested
    mass: Optional[np.ndarray] = None     # (N,)
    lbol: Optional[np.ndarray] = None     # (N,)
    mdust: Optional[np.ndarray] = None    # (N,)
    sfr: Optional[np.ndarray] = None      # (N,)
    emlines: Optional[np.ndarray] = None  # (N, nemline) if requested
    indx: Optional[np.ndarray] = None     # (N, nindx) if requested
    young: Optional[np.ndarray] = None    # (N, nspec) intrinsic young component
    old: Optional[np.ndarray] = None      # (N, nspec) intrinsic old component


@dataclass
class CSPBatchGradResult:
    """Outputs and gradients from an adjoint batched CSP evaluation."""
    forward: CSPBatchResult
    params_bar: Dict[str, np.ndarray]                   # name -> (N,) parameter gradients
    raw_params_bar: np.ndarray                          # (NBGRAD, N) raw gradient matrix
    weights_bar: Optional[np.ndarray] = None            # (N, ntfull) if ssp_weights was passed
    weights_z_bar: Optional[np.ndarray] = None          # (N, nz, ntfull) if ssp_weights_z was passed


class CSPBatch:
    """Manages parameter packing, forward execution, and analytical gradient evaluations for a batch of models."""

    def __init__(
        self,
        context: "Context",
        params: Optional[Dict[str, Union[float, Sequence[float], np.ndarray]]] = None,
        ssp_weights: Optional[np.ndarray] = None,
        ssp_weights_z: Optional[np.ndarray] = None,
        zred_mags: Optional[Union[Sequence[float], np.ndarray]] = None,
        **kwargs,
    ):
        self._context = context
        all_params = {}
        if params is not None:
            all_params.update(params)
        all_params.update(kwargs)

        self._ssp_weights = ssp_weights
        self._ssp_weights_z = ssp_weights_z
        self._zred_mags = zred_mags

        # Determine batch size N
        self.n_models = self._determine_n(all_params, ssp_weights, ssp_weights_z, zred_mags)
        self._packed_names, self._packed_values, self._val_keepalive = self._pack_params(all_params, self.n_models)
        self._w_f, self._wz_f = self._prepare_weights()

    def _determine_n(self, params, ssp_weights, ssp_weights_z, zred_mags) -> int:
        n = 1
        for k, v in params.items():
            if isinstance(v, (np.ndarray, list, tuple)):
                n = max(n, len(v))
        if ssp_weights is not None:
            sw = np.asarray(ssp_weights)
            # Accept (N, ntfull) or (ntfull, N)
            if sw.ndim == 2:
                if sw.shape[1] == self._context.ntfull:
                    n = max(n, sw.shape[0])
                elif sw.shape[0] == self._context.ntfull:
                    n = max(n, sw.shape[1])
        if ssp_weights_z is not None:
            swz = np.asarray(ssp_weights_z)
            if swz.ndim == 3:
                # Accept (N, nz, ntfull) or (ntfull, nz, N)
                if swz.shape[0] == self._context.ntfull:
                    n = max(n, swz.shape[2])
                else:
                    n = max(n, swz.shape[0])
        if zred_mags is not None:
            n = max(n, len(zred_mags))
        return n

    def _pack_params(self, params: Dict[str, Any], n: int):
        c_names = []
        c_values = []
        keepalive = []
        for name, val in params.items():
            val_arr = np.asarray(val, dtype=np.float64)
            if val_arr.ndim == 0 or len(val_arr) == 1:
                val_full = np.full(n, float(val_arr), dtype=np.float64)
            else:
                val_full = np.ascontiguousarray(val_arr, dtype=np.float64)
                if len(val_full) != n:
                    raise ValueError(f"Parameter '{name}' has length {len(val_full)}, expected {n}")

            keepalive.append(val_full)
            c_names.append(name.encode("utf-8"))
            c_values.append(val_full.ctypes.data_as(_c_api.POINTER(c_double)))

        names_arr = (c_char_p * len(c_names))(*c_names)
        vals_arr = (_c_api.POINTER(c_double) * len(c_values))(*c_values)
        return names_arr, vals_arr, keepalive

    def _prepare_weights(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        # ssp_weights must be column-major (ntfull, N)
        w_f = None
        if self._ssp_weights is not None:
            w = np.asarray(self._ssp_weights, dtype=np.float64)
            if w.shape == (self.n_models, self._context.ntfull):
                w_f = np.asfortranarray(w.T)
            elif w.shape == (self._context.ntfull, self.n_models):
                w_f = np.asfortranarray(w)
            else:
                raise ValueError(
                    f"ssp_weights shape {w.shape} invalid; expected ({self.n_models}, {self._context.ntfull})"
                )

        wz_f = None
        if self._ssp_weights_z is not None:
            wz = np.asarray(self._ssp_weights_z, dtype=np.float64)
            nt = self._context.ntfull
            nz = self._context.nz
            if wz.shape == (self.n_models, nt, nz):
                wz_f = np.asfortranarray(wz.transpose(1, 2, 0))
            elif wz.shape == (self.n_models, nz, nt):
                wz_f = np.asfortranarray(wz.transpose(2, 1, 0))
            elif wz.shape == (nt, nz, self.n_models):
                wz_f = np.asfortranarray(wz)
            else:
                raise ValueError(
                    f"ssp_weights_z shape {wz.shape} invalid; expected ({self.n_models}, {nt}, {nz}) or ({nt}, {nz}, {self.n_models})"
                )

        return w_f, wz_f

    def compute(
        self,
        compute_spec: bool = True,
        compute_emlines: bool = False,
        compute_indices: bool = False,
        parts: bool = False,
    ) -> CSPBatchResult:
        """Execute the forward batch computation."""
        n = self.n_models
        nbands = self._context.nbands
        nspec = self._context.nspec
        nemline = self._context.nemline
        nindx = self._context.nindx

        mags = np.empty((nbands, n), dtype=np.float64, order="F")
        mass = np.empty(n, dtype=np.float64)
        lbol = np.empty(n, dtype=np.float64)
        mdust = np.empty(n, dtype=np.float64)
        sfr = np.empty(n, dtype=np.float64)

        spec = np.empty((nspec, n), dtype=np.float64, order="F") if compute_spec or parts else None
        emlines = np.empty((nemline, n), dtype=np.float64, order="F") if compute_emlines else None
        indx = np.empty((nindx, n), dtype=np.float64, order="F") if compute_indices else None

        young = np.empty((nspec, n), dtype=np.float64, order="F") if parts else None
        old = np.empty((nspec, n), dtype=np.float64, order="F") if parts else None

        w_f, wz_f = self._prepare_weights()
        zred_f = None
        if self._zred_mags is not None:
            zred_f = np.ascontiguousarray(self._zred_mags, dtype=np.float64)

        w_ptr = w_f.ctypes.data_as(_c_api.POINTER(c_double)) if w_f is not None else None
        wz_ptr = wz_f.ctypes.data_as(_c_api.POINTER(c_double)) if wz_f is not None else None
        zred_ptr = zred_f.ctypes.data_as(_c_api.POINTER(c_double)) if zred_f is not None else None

        spec_ptr = spec.ctypes.data_as(_c_api.POINTER(c_double)) if spec is not None else None
        eml_ptr = emlines.ctypes.data_as(_c_api.POINTER(c_double)) if emlines is not None else None
        indx_ptr = indx.ctypes.data_as(_c_api.POINTER(c_double)) if indx is not None else None

        status = c_int(0)
        n_params = len(self._val_keepalive)

        if parts:
            mass_y = np.empty(n, dtype=np.float64)
            mass_o = np.empty(n, dtype=np.float64)
            lbol_y = np.empty(n, dtype=np.float64)
            lbol_o = np.empty(n, dtype=np.float64)
            _c_api._LIB.opensps_context_compute_csp_batch_parts(
                self._context.handle,
                n,
                n_params,
                self._packed_names,
                self._packed_values,
                w_ptr,
                mags.ctypes.data_as(_c_api.POINTER(c_double)),
                spec_ptr,
                mass.ctypes.data_as(_c_api.POINTER(c_double)),
                lbol.ctypes.data_as(_c_api.POINTER(c_double)),
                mdust.ctypes.data_as(_c_api.POINTER(c_double)),
                sfr.ctypes.data_as(_c_api.POINTER(c_double)),
                eml_ptr,
                indx_ptr,
                young.ctypes.data_as(_c_api.POINTER(c_double)),
                old.ctypes.data_as(_c_api.POINTER(c_double)),
                mass_y.ctypes.data_as(_c_api.POINTER(c_double)),
                mass_o.ctypes.data_as(_c_api.POINTER(c_double)),
                lbol_y.ctypes.data_as(_c_api.POINTER(c_double)),
                lbol_o.ctypes.data_as(_c_api.POINTER(c_double)),
                byref(status),
            )
        elif wz_f is not None or zred_f is not None:
            # Use grad entry point with NULL cotangents to forward wz / zred_mags
            _c_api._LIB.opensps_context_compute_csp_batch_grad(
                self._context.handle,
                n,
                n_params,
                self._packed_names,
                self._packed_values,
                w_ptr,
                wz_ptr,
                zred_ptr,
                None, None, None, None,
                mags.ctypes.data_as(_c_api.POINTER(c_double)),
                spec_ptr,
                mass.ctypes.data_as(_c_api.POINTER(c_double)),
                lbol.ctypes.data_as(_c_api.POINTER(c_double)),
                mdust.ctypes.data_as(_c_api.POINTER(c_double)),
                sfr.ctypes.data_as(_c_api.POINTER(c_double)),
                eml_ptr,
                indx_ptr,
                None, None, None,
                byref(status),
            )
        else:
            _c_api._LIB.opensps_context_compute_csp_batch(
                self._context.handle,
                n,
                n_params,
                self._packed_names,
                self._packed_values,
                w_ptr,
                mags.ctypes.data_as(_c_api.POINTER(c_double)),
                spec_ptr,
                mass.ctypes.data_as(_c_api.POINTER(c_double)),
                lbol.ctypes.data_as(_c_api.POINTER(c_double)),
                mdust.ctypes.data_as(_c_api.POINTER(c_double)),
                sfr.ctypes.data_as(_c_api.POINTER(c_double)),
                eml_ptr,
                indx_ptr,
                byref(status),
            )

        _c_api.check_status(status.value, "compute_csp_batch")

        return CSPBatchResult(
            mags=mags.T.copy(),
            spec=spec.T.copy() if spec is not None else None,
            mass=mass,
            lbol=lbol,
            mdust=mdust,
            sfr=sfr,
            emlines=emlines.T.copy() if emlines is not None else None,
            indx=indx.T.copy() if indx is not None else None,
            young=young.T.copy() if young is not None else None,
            old=old.T.copy() if old is not None else None,
        )

    def compute_gradients(
        self,
        mags_bar: Optional[np.ndarray] = None,
        spec_bar: Optional[np.ndarray] = None,
        mass_bar: Optional[np.ndarray] = None,
        lbol_bar: Optional[np.ndarray] = None,
        compute_spec: bool = False,
        compute_emlines: bool = False,
        compute_indices: bool = False,
    ) -> CSPBatchGradResult:
        """Execute the reverse-mode adjoint pass to compute analytical gradients."""
        n = self.n_models
        nbands = self._context.nbands
        nspec = self._context.nspec
        ntfull = self._context.ntfull
        nz = self._context.nz
        nemline = self._context.nemline
        nindx = self._context.nindx

        # Forward outputs
        mags = np.empty((nbands, n), dtype=np.float64, order="F")
        mass = np.empty(n, dtype=np.float64)
        lbol = np.empty(n, dtype=np.float64)
        mdust = np.empty(n, dtype=np.float64)
        sfr = np.empty(n, dtype=np.float64)
        spec = np.empty((nspec, n), dtype=np.float64, order="F") if compute_spec or (spec_bar is not None) else None
        emlines = np.empty((nemline, n), dtype=np.float64, order="F") if compute_emlines else None
        indx = np.empty((nindx, n), dtype=np.float64, order="F") if compute_indices else None

        # Adjoint inputs (cotangents)
        mb_f = None
        if mags_bar is not None:
            mb = np.asarray(mags_bar, dtype=np.float64)
            if mb.shape != (n, nbands):
                raise ValueError(f"mags_bar shape {mb.shape} does not match (n_models={n}, nbands={nbands})")
            mb_f = np.asfortranarray(mb.T)

        sb_f = None
        if spec_bar is not None:
            sb = np.asarray(spec_bar, dtype=np.float64)
            if sb.shape != (n, nspec):
                raise ValueError(f"spec_bar shape {sb.shape} does not match (n_models={n}, nspec={nspec})")
            sb_f = np.asfortranarray(sb.T)

        mass_b_c = None
        if mass_bar is not None:
            msb = np.asarray(mass_bar, dtype=np.float64)
            if msb.shape != (n,):
                raise ValueError(f"mass_bar shape {msb.shape} does not match (n_models={n},)")
            mass_b_c = np.ascontiguousarray(msb)

        lbol_b_c = None
        if lbol_bar is not None:
            lb = np.asarray(lbol_bar, dtype=np.float64)
            if lb.shape != (n,):
                raise ValueError(f"lbol_bar shape {lb.shape} does not match (n_models={n},)")
            lbol_b_c = np.ascontiguousarray(lb)

        # Adjoint outputs (gradients)
        raw_pbar = np.zeros((NBGRAD, n), dtype=np.float64, order="F")
        w_f, wz_f = self._w_f, self._wz_f
        w_bar = np.zeros((ntfull, n), dtype=np.float64, order="F") if w_f is not None else None
        wz_bar = np.zeros((ntfull, nz, n), dtype=np.float64, order="F") if wz_f is not None else None

        zred_f = np.ascontiguousarray(self._zred_mags, dtype=np.float64) if self._zred_mags is not None else None

        status = c_int(0)
        n_params = len(self._val_keepalive)

        _c_api._LIB.opensps_context_compute_csp_batch_grad(
            self._context.handle,
            n,
            n_params,
            self._packed_names,
            self._packed_values,
            w_f.ctypes.data_as(_c_api.POINTER(c_double)) if w_f is not None else None,
            wz_f.ctypes.data_as(_c_api.POINTER(c_double)) if wz_f is not None else None,
            zred_f.ctypes.data_as(_c_api.POINTER(c_double)) if zred_f is not None else None,
            mb_f.ctypes.data_as(_c_api.POINTER(c_double)) if mb_f is not None else None,
            sb_f.ctypes.data_as(_c_api.POINTER(c_double)) if sb_f is not None else None,
            mass_b_c.ctypes.data_as(_c_api.POINTER(c_double)) if mass_b_c is not None else None,
            lbol_b_c.ctypes.data_as(_c_api.POINTER(c_double)) if lbol_b_c is not None else None,
            mags.ctypes.data_as(_c_api.POINTER(c_double)),
            spec.ctypes.data_as(_c_api.POINTER(c_double)) if spec is not None else None,
            mass.ctypes.data_as(_c_api.POINTER(c_double)),
            lbol.ctypes.data_as(_c_api.POINTER(c_double)),
            mdust.ctypes.data_as(_c_api.POINTER(c_double)),
            sfr.ctypes.data_as(_c_api.POINTER(c_double)),
            emlines.ctypes.data_as(_c_api.POINTER(c_double)) if emlines is not None else None,
            indx.ctypes.data_as(_c_api.POINTER(c_double)) if indx is not None else None,
            raw_pbar.ctypes.data_as(_c_api.POINTER(c_double)),
            w_bar.ctypes.data_as(_c_api.POINTER(c_double)) if w_bar is not None else None,
            wz_bar.ctypes.data_as(_c_api.POINTER(c_double)) if wz_bar is not None else None,
            byref(status),
        )
        _c_api.check_status(status.value, "compute_csp_batch_grad")

        # Map raw parameter gradients to dictionary
        pbar_dict = {}
        for row_idx in range(NBGRAD):
            pname = BGRAD_TO_PARAM.get(BGRAD(row_idx))
            if pname:
                pbar_dict[pname] = raw_pbar[row_idx, :].copy()

        fwd = CSPBatchResult(
            mags=mags.T.copy(),
            spec=spec.T.copy() if spec is not None else None,
            mass=mass,
            lbol=lbol,
            mdust=mdust,
            sfr=sfr,
            emlines=emlines.T.copy() if emlines is not None else None,
            indx=indx.T.copy() if indx is not None else None,
        )

        return CSPBatchGradResult(
            forward=fwd,
            params_bar=pbar_dict,
            raw_params_bar=raw_pbar.copy(),
            weights_bar=w_bar.T.copy() if w_bar is not None else None,
            weights_z_bar=wz_bar.transpose(2, 0, 1).copy() if wz_bar is not None else None,
        )
