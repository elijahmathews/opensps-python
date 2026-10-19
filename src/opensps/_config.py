"""Where the core library finds its data: the one place in the package that knows about it.

The library (``fsps_resolve_paths`` in the core) takes the first of these that holds a data
tree (a ``data/`` directory with ``allfilters.dat`` or ``FILTER_LIST``):

1. ``$SPS_HOME``
2. ``$FSPS_DATA_HOME``
3. ``$XDG_DATA_HOME/fsps``
4. ``~/.local/share/fsps``
5. ``/usr/share/fsps``, ``/usr/local/share/fsps``

and stops the process if none does. This module mirrors that search so that Python can raise
an exception instead, and lets a program choose the data explicitly
(:func:`set_data_home`, or ``$OPENSPS_DATA_HOME``): the choice is passed to the library by
setting both ``SPS_HOME`` and ``FSPS_DATA_HOME`` before a context is set up (the library reads
them at setup).

When the core moves to its new data format, only this module changes.
"""

import os
from pathlib import Path

__all__ = ["set_data_home", "get_data_home", "find_data_home", "DataNotFoundError"]

_ENV_OVERRIDE = "OPENSPS_DATA_HOME"
_data_home = None


class DataNotFoundError(FileNotFoundError):
    """No FSPS data tree was found."""


def _has_data(path):
    p = Path(path) / "data"
    return (p / "allfilters.dat").is_file() or (p / "FILTER_LIST").is_file()


def set_data_home(path):
    """Use the data tree at ``path`` (the directory that contains ``data/``) for contexts set
    up from now on. ``None`` returns to the library's own search."""
    global _data_home
    if path is None:
        _data_home = None
        return
    path = Path(path).expanduser().resolve()
    if not _has_data(path):
        raise DataNotFoundError(f"no FSPS data tree under {path} (expected {path}/data/allfilters.dat)")
    _data_home = path


def get_data_home():
    """The explicitly chosen data root (:func:`set_data_home` or ``$OPENSPS_DATA_HOME``), or
    ``None``."""
    if _data_home is not None:
        return _data_home
    env = os.environ.get(_ENV_OVERRIDE)
    return Path(env).expanduser().resolve() if env else None


def _library_search():
    """The library's own search order (see the module docstring)."""
    for var in ("SPS_HOME", "FSPS_DATA_HOME"):
        v = os.environ.get(var)
        if v:
            yield Path(v)
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        yield Path(xdg) / "fsps"
    home = os.environ.get("HOME")
    if home:
        yield Path(home) / ".local" / "share" / "fsps"
    yield Path("/usr/share/fsps")
    yield Path("/usr/local/share/fsps")


def find_data_home(explicit=None):
    """The data root a context set up now would use."""
    chosen = Path(explicit).expanduser().resolve() if explicit else get_data_home()
    if chosen is not None:
        if not _has_data(chosen):
            raise DataNotFoundError(f"no FSPS data tree under {chosen}")
        return chosen
    for p in _library_search():
        if _has_data(p):
            return p
    raise DataNotFoundError(
        "FSPS data not found: call opensps.set_data_home(path) or set OPENSPS_DATA_HOME "
        "(or SPS_HOME / FSPS_DATA_HOME) to the directory that contains data/")


def prepare_environment(explicit=None):
    """Called before every context setup: check that data exist and point the library at the
    chosen tree. Returns the data root."""
    root = find_data_home(explicit)
    if explicit is not None or get_data_home() is not None:
        os.environ["SPS_HOME"] = str(root)
        os.environ["FSPS_DATA_HOME"] = str(root)
    return root
