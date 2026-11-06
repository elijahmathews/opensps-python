"""OpenSPS: Modernized, High-Throughput Stellar Population Synthesis.

A high-performance Python interface to libopensps, featuring:
- High-throughput batched CSP calculation on GPU and CPU.
- Batched SSP operators for fast IMF variations.
- Reverse-mode analytical gradients for 33 physical parameters and SFH weights.
- PyTorch autograd integration for end-to-end differentiable modeling.
- Drop-in StellarPopulation compatibility with python-fsps for Prospector.
"""

from opensps._version import (
    __version__,
    __libopensps_c_api_supported__,
)
from opensps.constants import (
    BGRAD,
    PARAM_TO_BGRAD,
    BGRAD_TO_PARAM,
    NBGRAD,
    OpenSPSError,
    FeatureNotSupportedError,
    IncompatibleLibraryError,
)
from opensps._c_api import (
    get_driver_version,
    get_nbgrad,
    get_grad_param_name,
    HAS_BATCH_GRAD,
    HAS_SSP_OPS,
    HAS_DEVICE_PTR,
)
from opensps.context import Context
from opensps.batch import CSPBatch, CSPBatchResult, CSPBatchGradResult
from opensps.ssp import SSPOperator
from opensps.fsps import StellarPopulation

__all__ = [
    "Context",
    "CSPBatch",
    "CSPBatchResult",
    "CSPBatchGradResult",
    "SSPOperator",
    "StellarPopulation",
    "BGRAD",
    "PARAM_TO_BGRAD",
    "BGRAD_TO_PARAM",
    "NBGRAD",
    "OpenSPSError",
    "FeatureNotSupportedError",
    "IncompatibleLibraryError",
    "get_driver_version",
    "get_nbgrad",
    "get_grad_param_name",
    "HAS_BATCH_GRAD",
    "HAS_SSP_OPS",
    "HAS_DEVICE_PTR",
    "__version__",
    "__libopensps_c_api_supported__",
]
