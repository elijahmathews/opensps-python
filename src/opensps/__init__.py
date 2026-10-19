"""opensps: Python bindings for libopensps, a modernized FSPS.

- :class:`Context`: the library's context API (parameters by name, SSP grids, single CSPs,
  the batched CSP with one array per parameter); NumPy arrays in and out.
- :class:`StellarPopulation`: a python-fsps compatible layer on top of it.
"""

__version__ = "0.1.0.dev0"

from ._config import DataNotFoundError, find_data_home, get_data_home, set_data_home
from ._core import (
    Context,
    DeviceSpectra,
    FSPSError,
    UnknownParameterError,
    api_version,
    header_api_version,
)
from .compat import StellarPopulation, install_as_fsps
from .filters import emline_names, find_filter, get_filter, list_filters

__all__ = [
    "Context", "DeviceSpectra", "FSPSError", "UnknownParameterError", "StellarPopulation",
    "DataNotFoundError", "set_data_home", "get_data_home", "find_data_home",
    "list_filters", "find_filter", "get_filter", "emline_names", "install_as_fsps",
    "api_version", "header_api_version",
]

# The header these bindings were compiled against must match the loaded library: same major
# version, and the library at least as new
_lib, _hdr = api_version(), header_api_version()
if _lib[0] != _hdr[0] or _lib[1] < _hdr[1]:
    raise ImportError(f"opensps was built against the C API {_hdr} but the loaded core library "
                      f"provides {_lib}")
del _lib, _hdr
