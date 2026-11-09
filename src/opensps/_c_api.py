"""Low-level ctypes bindings to libopensps.so."""

import ctypes
from ctypes import (
    c_int,
    c_double,
    c_char_p,
    c_void_p,
    POINTER,
    byref,
    create_string_buffer,
)
import os
import sys
from typing import Optional, Tuple
from opensps.constants import OpenSPSError, FeatureNotSupportedError, IncompatibleLibraryError
from opensps._version import (
    __version__,
    __libopensps_c_api_supported__,
    is_c_api_compatible,
)

# -----------------------------------------------------------------------------
# Library Loader
# -----------------------------------------------------------------------------

def _find_library() -> str:
    # 1. Environment variable override
    env_path = os.environ.get("OPENSPS_LIB_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path
    env_dir = os.environ.get("OPENSPS_LIB_DIR")
    if env_dir:
        candidate = os.path.join(env_dir, "libopensps.so")
        if os.path.isfile(candidate):
            return candidate

    # 2. Check development workspace relative to this file
    here = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        # In sibling libopensps build dir
        os.path.abspath(os.path.join(here, "../../../../libopensps/build/libopensps.so")),
        os.path.abspath(os.path.join(here, "../../../libopensps/build/libopensps.so")),
        os.path.abspath(os.path.join(here, "../../libopensps/build/libopensps.so")),
        os.path.abspath(os.path.join(here, "libopensps.so")),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c

    # 3. System library resolution
    found = ctypes.util.find_library("opensps")
    if found:
        return found

    for std in ["/usr/local/lib/libopensps.so", "/usr/lib64/libopensps.so", "/usr/lib/libopensps.so"]:
        if os.path.isfile(std):
            return std

    raise ImportError(
        "Could not find libopensps.so. Please ensure libopensps is compiled and set "
        "OPENSPS_LIB_PATH or OPENSPS_LIB_DIR."
    )


_LIB = ctypes.CDLL(_find_library())

# -----------------------------------------------------------------------------
# Driver Utilities & Version Negotiation
# -----------------------------------------------------------------------------

_DRIVER_VERSION: Optional[Tuple[int, int, int]] = None
if hasattr(_LIB, "opensps_get_driver_version"):
    _LIB.opensps_get_driver_version.argtypes = [POINTER(c_int), POINTER(c_int), POINTER(c_int)]
    _LIB.opensps_get_driver_version.restype = None

    maj, min_, pat = c_int(0), c_int(0), c_int(0)
    _LIB.opensps_get_driver_version(byref(maj), byref(min_), byref(pat))
    _DRIVER_VERSION = (maj.value, min_.value, pat.value)

    if not is_c_api_compatible(_DRIVER_VERSION):
        raise IncompatibleLibraryError(
            f"Loaded libopensps C API version {_DRIVER_VERSION[0]}.{_DRIVER_VERSION[1]}.{_DRIVER_VERSION[2]} "
            f"is incompatible with opensps-python {__version__} "
            f"(requires C API {__libopensps_c_api_supported__})."
        )
else:
    raise IncompatibleLibraryError(
        f"Loaded libopensps does not export 'opensps_get_driver_version'. "
        f"opensps-python {__version__} requires C API {__libopensps_c_api_supported__}."
    )

# Dynamic Capability Negotiation (Feature Probing)
HAS_BATCH_GRAD = hasattr(_LIB, "opensps_context_compute_csp_batch_grad") and hasattr(_LIB, "opensps_get_nbgrad")
HAS_SSP_OPS = hasattr(_LIB, "opensps_context_build_ssp_ops") and hasattr(_LIB, "opensps_context_evaluate_ssp_op")
HAS_DEVICE_PTR = hasattr(_LIB, "opensps_context_batch_spec_device")
HAS_MAG_COMPUTE = hasattr(_LIB, "opensps_context_set_mag_compute")
if HAS_MAG_COMPUTE:
    _LIB.opensps_context_set_mag_compute.argtypes = [c_int, c_int, POINTER(c_int), POINTER(c_int)]
    _LIB.opensps_context_set_mag_compute.restype = None

if HAS_BATCH_GRAD:
    _LIB.opensps_get_nbgrad.argtypes = [POINTER(c_int)]
    _LIB.opensps_get_nbgrad.restype = None

    _LIB.opensps_get_grad_param_name.argtypes = [c_int, c_char_p, c_int, POINTER(c_int)]
    _LIB.opensps_get_grad_param_name.restype = None

_LIB.opensps_get_last_error.argtypes = [POINTER(c_int), c_char_p, c_int]
_LIB.opensps_get_last_error.restype = None

_LIB.opensps_clear_error.argtypes = []
_LIB.opensps_clear_error.restype = None

# Context Lifecycle
_LIB.opensps_context_create.argtypes = [POINTER(c_int), POINTER(c_int)]
_LIB.opensps_context_create.restype = None

_LIB.opensps_context_destroy.argtypes = [c_int, POINTER(c_int)]
_LIB.opensps_context_destroy.restype = None

_LIB.opensps_context_setup.argtypes = [c_int, c_char_p, c_char_p, c_char_p, c_int, POINTER(c_int)]
_LIB.opensps_context_setup.restype = None

# Context Parameters
_LIB.opensps_context_set_int.argtypes = [c_int, c_char_p, c_int, POINTER(c_int)]
_LIB.opensps_context_set_int.restype = None

_LIB.opensps_context_set_float.argtypes = [c_int, c_char_p, c_double, POINTER(c_int)]
_LIB.opensps_context_set_float.restype = None

_LIB.opensps_context_set_str.argtypes = [c_int, c_char_p, c_char_p, POINTER(c_int)]
_LIB.opensps_context_set_str.restype = None

_LIB.opensps_context_set_fast_mode.argtypes = [c_int, c_int, POINTER(c_int)]
_LIB.opensps_context_set_fast_mode.restype = None

# Dimensions
_LIB.opensps_context_get_dims.argtypes = [c_int, POINTER(c_int), POINTER(c_int), POINTER(c_int)]
_LIB.opensps_context_get_dims.restype = None

_LIB.opensps_context_get_nspec.argtypes = [c_int, POINTER(c_int), POINTER(c_int)]
_LIB.opensps_context_get_nspec.restype = None

_LIB.opensps_context_get_ntfull.argtypes = [c_int, POINTER(c_int), POINTER(c_int)]
_LIB.opensps_context_get_ntfull.restype = None

_LIB.opensps_context_get_nbands.argtypes = [c_int, POINTER(c_int), POINTER(c_int)]
_LIB.opensps_context_get_nbands.restype = None

_LIB.opensps_context_get_nindx.argtypes = [c_int, POINTER(c_int), POINTER(c_int)]
_LIB.opensps_context_get_nindx.restype = None

_LIB.opensps_context_get_nz.argtypes = [c_int, POINTER(c_int), POINTER(c_int)]
_LIB.opensps_context_get_nz.restype = None

_LIB.opensps_context_get_nemline.argtypes = [c_int, POINTER(c_int), POINTER(c_int)]
_LIB.opensps_context_get_nemline.restype = None

_LIB.opensps_context_get_zsol.argtypes = [c_int, POINTER(c_double), POINTER(c_int)]
_LIB.opensps_context_get_zsol.restype = None

# Grids
_LIB.opensps_context_get_lambda.argtypes = [c_int, POINTER(c_double), POINTER(c_int)]
_LIB.opensps_context_get_lambda.restype = None

_LIB.opensps_context_get_ages.argtypes = [c_int, POINTER(c_double), POINTER(c_int)]
_LIB.opensps_context_get_ages.restype = None

_LIB.opensps_context_get_zlegend.argtypes = [c_int, POINTER(c_double), POINTER(c_int)]
_LIB.opensps_context_get_zlegend.restype = None

_LIB.opensps_context_get_emlambda.argtypes = [c_int, POINTER(c_double), POINTER(c_int)]
_LIB.opensps_context_get_emlambda.restype = None

_LIB.opensps_context_get_filter_data.argtypes = [
    c_int,
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_int),
]
_LIB.opensps_context_get_filter_data.restype = None

# Context Computations
_LIB.opensps_context_compute_ssp.argtypes = [
    c_int,
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_int),
]
_LIB.opensps_context_compute_ssp.restype = None

_LIB.opensps_context_set_sfh_tab.argtypes = [
    c_int,
    c_int,
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_int),
]
_LIB.opensps_context_set_sfh_tab.restype = None

_LIB.opensps_context_compute_csp.argtypes = [c_int, c_int, POINTER(c_int), POINTER(c_int)]
_LIB.opensps_context_compute_csp.restype = None

_LIB.opensps_context_get_csp.argtypes = [
    c_int,
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_int),
]
_LIB.opensps_context_get_csp.restype = None

# Batched Computations
_LIB.opensps_context_compute_csp_batch.argtypes = [
    c_int,
    c_int,
    c_int,
    POINTER(c_char_p),
    POINTER(POINTER(c_double)),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_int),
]
_LIB.opensps_context_compute_csp_batch.restype = None

_LIB.opensps_context_compute_csp_batch_parts.argtypes = [
    c_int,
    c_int,
    c_int,
    POINTER(c_char_p),
    POINTER(POINTER(c_double)),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_int),
]
_LIB.opensps_context_compute_csp_batch_parts.restype = None

if HAS_BATCH_GRAD:
    _LIB.opensps_context_compute_csp_batch_grad.argtypes = [
        c_int,
        c_int,
        c_int,
        POINTER(c_char_p),
        POINTER(POINTER(c_double)),
        POINTER(c_double),  # ssp_weights
        POINTER(c_double),  # ssp_weights_z
        POINTER(c_double),  # zred_mags
        POINTER(c_double),  # mags_bar
        POINTER(c_double),  # spec_bar
        POINTER(c_double),  # mass_bar
        POINTER(c_double),  # lbol_bar
        POINTER(c_double),  # mags
        POINTER(c_double),  # spec
        POINTER(c_double),  # mass
        POINTER(c_double),  # lbol
        POINTER(c_double),  # mdust
        POINTER(c_double),  # sfr
        POINTER(c_double),  # emlines
        POINTER(c_double),  # indx
        POINTER(c_double),  # params_bar
        POINTER(c_double),  # weights_bar
        POINTER(c_double),  # weights_z_bar
        POINTER(c_int),     # status
    ]
    _LIB.opensps_context_compute_csp_batch_grad.restype = None

# Device Spectra
if HAS_DEVICE_PTR:
    _LIB.opensps_context_batch_spec_device.argtypes = [
        c_int,
        POINTER(c_void_p),
        POINTER(c_int),
        POINTER(c_int),
        POINTER(c_int),
        POINTER(c_int),
    ]
    _LIB.opensps_context_batch_spec_device.restype = None

if hasattr(_LIB, "opensps_context_batch_spec_host"):
    _LIB.opensps_context_batch_spec_host.argtypes = [
        c_int,
        POINTER(c_void_p),
        POINTER(c_int),
        POINTER(c_int),
        POINTER(c_int),
        POINTER(c_int),
    ]
    _LIB.opensps_context_batch_spec_host.restype = None

# SSP Operators
if HAS_SSP_OPS:
    _LIB.opensps_context_build_ssp_ops.argtypes = [c_int, c_int, POINTER(c_int)]
    _LIB.opensps_context_build_ssp_ops.restype = None

    _LIB.opensps_context_free_ssp_ops.argtypes = [c_int, POINTER(c_int)]
    _LIB.opensps_context_free_ssp_ops.restype = None

    _LIB.opensps_context_has_ssp_ops.argtypes = [c_int, POINTER(c_int), POINTER(c_int)]
    _LIB.opensps_context_has_ssp_ops.restype = None

    _LIB.opensps_context_evaluate_ssp_op.argtypes = [
        c_int,
        c_int,
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_int),
    ]
    _LIB.opensps_context_evaluate_ssp_op.restype = None


# -----------------------------------------------------------------------------
# Python Helpers
# -----------------------------------------------------------------------------

def check_status(status: int, operation: str = ""):
    """Check a returned OpenSPS status code and raise OpenSPSError if non-zero."""
    if status != 0:
        err_msg = create_string_buffer(256)
        last_st = c_int(0)
        _LIB.opensps_get_last_error(byref(last_st), err_msg, 256)
        msg = err_msg.value.decode("utf-8", errors="replace").strip()
        prefix = f"{operation}: " if operation else ""
        raise OpenSPSError(status, f"{prefix}{msg}")


def get_driver_version() -> Tuple[int, int, int]:
    if _DRIVER_VERSION is not None:
        return _DRIVER_VERSION
    major = c_int(0)
    minor = c_int(0)
    patch = c_int(0)
    _LIB.opensps_get_driver_version(byref(major), byref(minor), byref(patch))
    return (major.value, minor.value, patch.value)


def get_nbgrad() -> int:
    if not HAS_BATCH_GRAD:
        raise FeatureNotSupportedError(
            "Analytical gradients require a libopensps shared library built with batch gradient support."
        )
    nb = c_int(0)
    _LIB.opensps_get_nbgrad(byref(nb))
    return nb.value


def get_grad_param_name(grad_idx: int) -> str:
    if not HAS_BATCH_GRAD:
        raise FeatureNotSupportedError(
            "Analytical gradients require a libopensps shared library built with batch gradient support."
        )
    buf = create_string_buffer(64)
    status = c_int(0)
    _LIB.opensps_get_grad_param_name(grad_idx, buf, 64, byref(status))
    check_status(status.value, f"get_grad_param_name({grad_idx})")
    return buf.value.decode("utf-8").strip()
