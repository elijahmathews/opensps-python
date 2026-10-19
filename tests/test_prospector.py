"""Optional: Prospector's source classes on opensps (skipped unless Prospector is importable).

``install_as_fsps()`` makes ``import fsps`` inside Prospector return opensps; the spectra of
CSPSpecBasis and FastStepBasis must equal what StellarPopulation gives for the same inputs.
"""

import sys

import numpy as np
import pytest

import opensps
from conftest import assert_identical


@pytest.fixture(scope="module")
def prospect_sources(data_home):
    saved = sys.modules.get("fsps")
    opensps.install_as_fsps()
    try:
        from prospect.sources import galaxy_basis
    except Exception as e:  # not installed, or its dependencies are missing
        pytest.skip(f"Prospector not importable: {e}")
    galaxy_basis.fsps = sys.modules["fsps"]
    yield galaxy_basis
    if saved is None:
        sys.modules.pop("fsps", None)
    else:
        sys.modules["fsps"] = saved


def test_cspspecbasis(prospect_sources):
    sps = prospect_sources.CSPSpecBasis(zcontinuous=1)
    params = dict(mass=np.array([1e10]), logzsol=np.array([-0.3]), dust2=np.array([0.4]),
                  tage=np.array([6.0]), tau=np.array([2.0]), sfh=np.array([4]),
                  add_neb_emission=True)
    wave, spec, mfrac = sps.get_galaxy_spectrum(**params)
    sp = opensps.StellarPopulation(zcontinuous=1, logzsol=-0.3, dust2=0.4, tau=2.0, sfh=4,
                                   add_neb_emission=True)
    w, s = sp.get_spectrum(tage=6.0)
    assert_identical(wave, w)
    np.testing.assert_allclose(spec, s, rtol=1e-14, atol=0)
    np.testing.assert_allclose(mfrac, sp.stellar_mass, rtol=1e-14)


def test_faststepbasis(prospect_sources):
    sps = prospect_sources.FastStepBasis(zcontinuous=1)
    agebins = np.array([[0.0, 8.0], [8.0, 9.0], [9.0, 10.0]])
    mass = np.array([1e8, 1e9, 5e9])
    wave, spec, mfrac = sps.get_galaxy_spectrum(agebins=agebins, mass=mass,
                                                logzsol=np.array([-0.5]))
    assert spec.shape == wave.shape and np.all(np.isfinite(spec)) and spec.max() > 0
    assert 0.3 < mfrac < 1.0
    # The same tabular SFH through StellarPopulation directly
    time, sfr, tmax = sps.convert_sfh(agebins, mass)
    sp = opensps.StellarPopulation(zcontinuous=1, logzsol=-0.5, sfh=3)
    sp.set_tabular_sfh(time, sfr)
    w, s = sp.get_spectrum(tage=tmax)
    np.testing.assert_allclose(spec, s / mass.sum(), rtol=1e-14, atol=0)
