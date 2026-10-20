"""Filter and emission-line names, read from the data tree the library uses.

The library itself knows filters and lines only by index; their names live in
``data/FILTER_LIST`` and ``data/emlines_info.dat`` (as for python-fsps).
"""

from functools import lru_cache
from pathlib import Path

from . import _config

__all__ = ["Filter", "list_filters", "find_filter", "get_filter", "emline_names"]


class Filter:
    """One band: ``index`` (0-based position in the magnitude arrays), ``name`` (lower case),
    ``fullname`` (the description in FILTER_LIST)."""

    def __init__(self, index, name, fullname):
        self.index = index
        self.name = name.lower()
        self.fullname = fullname

    def __repr__(self):
        return f"<Filter({self.name})>"


def _root(data_home=None):
    return Path(data_home) if data_home is not None else _config.find_data_home()


@lru_cache(maxsize=8)
def _filters(root):
    out = {}
    with open(Path(root) / "data" / "FILTER_LIST") as f:
        for line in f:
            parts = line.split(None, 2)
            if len(parts) < 2 or not parts[0].isdigit():
                continue
            flt = Filter(int(parts[0]) - 1, parts[1], parts[2].strip() if len(parts) > 2 else "")
            out[flt.name] = flt
    return out


def filters(data_home=None):
    """Mapping of lower-case filter name to :class:`Filter`, in index order."""
    return _filters(str(_root(data_home)))


def list_filters(data_home=None):
    """Filter names in index order (the order of the magnitude arrays)."""
    return list(filters(data_home))


def get_filter(name, data_home=None):
    """The :class:`Filter` of a name (case-insensitive)."""
    try:
        return filters(data_home)[name.lower()]
    except KeyError:
        raise KeyError(f"unknown filter {name!r}; see opensps.list_filters()") from None


def find_filter(text, data_home=None):
    """Names of the filters whose name or description contains ``text`` (case-insensitive)."""
    t = text.lower()
    return [k for k, v in filters(data_home).items() if t in k or t in v.fullname.lower()]


def band_indices(bands, data_home=None):
    """Indices of the named bands, in the given order."""
    f = filters(data_home)
    try:
        return [f[b.lower()].index for b in bands]
    except KeyError as e:
        raise KeyError(f"unknown filter {e.args[0]!r}; see opensps.list_filters()") from None


@lru_cache(maxsize=8)
def _emlines(root):
    names = []
    with open(Path(root) / "data" / "emlines_info.dat") as f:
        for line in f:
            if "," in line:
                names.append(line.split(",", 1)[1].strip())
    return tuple(names)


def emline_names(data_home=None):
    """Names of the emission lines, in the order of the line arrays."""
    return list(_emlines(str(_root(data_home))))
