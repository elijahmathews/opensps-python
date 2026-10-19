import os

import numpy as np
import pytest

import opensps

LIBS = ("mist", "miles", "DL07")


def _data_home():
    for var in ("OPENSPS_DATA_HOME", "FSPS_DATA_HOME"):
        if os.environ.get(var):
            return os.environ[var]
    try:
        return str(opensps.find_data_home())
    except opensps.DataNotFoundError:
        return None


DATA_HOME = _data_home()


def pytest_collection_modifyitems(config, items):
    if DATA_HOME is None:
        skip = pytest.mark.skip(reason="no FSPS data (set OPENSPS_DATA_HOME)")
        for item in items:
            item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def data_home():
    if DATA_HOME is not None:
        opensps.set_data_home(DATA_HOME)
    return DATA_HOME


@pytest.fixture(scope="session")
def ctx(data_home):
    """A context shared by tests that only read grids."""
    return opensps.Context(*LIBS)


@pytest.fixture
def fresh(data_home):
    """A context of its own (parameters start from the library defaults)."""
    c = opensps.Context(*LIBS)
    yield c
    c.close()


@pytest.fixture(scope="session")
def cref(data_home):
    """Direct ctypes calls into the same C library (reference path)."""
    from _cref import CRef

    return CRef()


def assert_identical(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape
    assert np.array_equal(a, b, equal_nan=True), f"max |diff| {np.nanmax(np.abs(a - b))}"
