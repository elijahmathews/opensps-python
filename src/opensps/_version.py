"""Version and compatibility metadata for opensps-python."""

__version__ = "0.1.0"

# Stated compatibility range for libopensps releases
__libopensps_supported__ = ">=0.1.0,<0.3.0"

# C API (driver) semantic version boundaries required by this binding
# libopensps exposes OPENSPS_API_VERSION (major=2, minor>=0)
MIN_C_API_VERSION = (2, 0, 0)
MAX_C_API_VERSION = (3, 0, 0)
