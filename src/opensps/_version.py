"""Version and compatibility metadata for opensps-python."""

__version__ = "0.1.0"

# C API version boundaries required by this binding (single source of truth)
MIN_C_API_VERSION = (0, 1, 0)
MAX_C_API_VERSION = (0, 3, 0)

# Derived range specifier for human-readable reporting and packaging metadata
__libopensps_c_api_supported__ = (
    f">={MIN_C_API_VERSION[0]}.{MIN_C_API_VERSION[1]}.{MIN_C_API_VERSION[2]},"
    f"<{MAX_C_API_VERSION[0]}.{MAX_C_API_VERSION[1]}.{MAX_C_API_VERSION[2]}"
)


def is_c_api_compatible(version: tuple[int, int, int]) -> bool:
    """Check whether a loaded C API driver version tuple is supported."""
    return MIN_C_API_VERSION <= version < MAX_C_API_VERSION

