"""Version and compatibility metadata for opensps-python."""

__version__ = "0.1.0"

# Stated compatibility range for libopensps C API specification.
# Under SemVer pre-1.0 (0.y.z), 0.2.0 may introduce breaking changes,
# so we conservatively support the 0.1.x series until 0.2.0 is validated.
__libopensps_c_api_supported__ = ">=0.1.0,<0.2.0"
__libopensps_supported__ = __libopensps_c_api_supported__

# C API (driver) semantic version boundaries required by this binding
# libopensps exposes OPENSPS_API_VERSION matching meson.build version (0.1.0)
MIN_C_API_VERSION = (0, 1, 0)
MAX_C_API_VERSION = (0, 2, 0)
