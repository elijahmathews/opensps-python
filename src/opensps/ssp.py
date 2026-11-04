"""Batched SSP operator wrapper for OpenSPS."""

import numpy as np
from ctypes import c_int, c_double, byref
from typing import Optional, Sequence, Tuple
from opensps import _c_api
from opensps.constants import OpenSPSError


class SSPOperator:
    """Represents an active SSP operator on an OpenSPS Context.

    The SSP operator decouples stellar isochrones from IMF integrals, allowing
    rapid evaluation across thousands of custom IMF slopes and parameters without
    recomputing isochrones or regenerating grids.
    """

    def __init__(self, context: "Context", all_metallicities: bool = False):
        self._context = context
        self.all_metallicities = all_metallicities
        self._build()

    def _build(self):
        status = c_int(0)
        _c_api._LIB.opensps_context_build_ssp_ops(
            self._context.handle,
            1 if self.all_metallicities else 0,
            byref(status),
        )
        _c_api.check_status(status.value, "build_ssp_ops")

    def free(self):
        """Release the operator resources from the context."""
        if self._context.handle:
            status = c_int(0)
            _c_api._LIB.opensps_context_free_ssp_ops(self._context.handle, byref(status))
            _c_api.check_status(status.value, "free_ssp_ops")

    def evaluate(
        self,
        imf_type: Optional[int] = None,
        imf_params: Optional[Sequence[float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Directly evaluate the SSP grid from the operator for an IMF.

        Parameters
        ----------
        imf_type : int, optional
            IMF type (0: Salpeter, 1: Chabrier, 2: Kroupa, 3: van Dokkum, 4: Dave).
            If omitted or None, uses the context's current imf_type.
        imf_params : sequence of 5 floats, optional
            Core IMF parameters: [imf1, imf2, imf3, vdmc, mdave].
            If omitted or None, uses context values.

        Returns
        -------
        mass : ndarray of shape (ntfull,)
            Surviving stellar mass per solar mass formed.
        lbol : ndarray of shape (ntfull,)
            Log10 bolometric luminosity in L_sun.
        spec : ndarray of shape (ntfull, nspec)
            SSP spectral grid in L_sun / Hz.
        """
        nt = self._context.ntfull
        nspec = self._context.nspec

        mass = np.empty(nt, dtype=np.float64, order="F")
        lbol = np.empty(nt, dtype=np.float64, order="F")
        # Column-major in Fortran: [nspec, nt]
        spec_c = np.empty((nspec, nt), dtype=np.float64, order="F")

        p_arr = None
        p_ptr = None
        if imf_params is not None:
            p_arr = np.ascontiguousarray(imf_params, dtype=np.float64)
            if len(p_arr) < 5:
                padded = np.zeros(5, dtype=np.float64)
                padded[:len(p_arr)] = p_arr
                p_arr = padded
            p_ptr = p_arr.ctypes.data_as(_c_api.POINTER(c_double))

        it = -1 if imf_type is None else int(imf_type)
        status = c_int(0)

        _c_api._LIB.opensps_context_evaluate_ssp_op(
            self._context.handle,
            it,
            p_ptr,
            mass.ctypes.data_as(_c_api.POINTER(c_double)),
            lbol.ctypes.data_as(_c_api.POINTER(c_double)),
            spec_c.ctypes.data_as(_c_api.POINTER(c_double)),
            byref(status),
        )
        _c_api.check_status(status.value, "evaluate_ssp_op")

        # Return spec transposed to standard (ntfull, nspec)
        return mass, lbol, spec_c.T.copy()
