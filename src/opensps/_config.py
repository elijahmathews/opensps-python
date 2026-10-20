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
setting both ``SPS_HOME`` and ``FSPS_DATA_HOME`` during a context's setup (the library reads
them there) and restoring them afterwards.

When the core moves to its new data format, only this module changes.
"""

import os
from contextlib import contextmanager
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


@contextmanager
def library_environment(explicit=None):
    """Around a context setup: check that data exist and, if a data root was chosen, point the
    library at it through ``SPS_HOME`` and ``FSPS_DATA_HOME``, restored afterwards (other code
    in the process, e.g. Prospector, reads ``SPS_HOME`` for its own legacy data)."""
    root = find_data_home(explicit)
    if explicit is None and get_data_home() is None:
        yield root
        return
    saved = {k: os.environ.get(k) for k in ("SPS_HOME", "FSPS_DATA_HOME")}
    os.environ["SPS_HOME"] = str(root)
    os.environ["FSPS_DATA_HOME"] = str(root)
    try:
        yield root
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
