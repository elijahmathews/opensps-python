"""StellarPopulation reproduces direct Context calls and follows python-fsps semantics."""

import sys

import numpy as np
import pytest

import opensps
from conftest import LIBS, assert_identical


@pytest.fixture(scope="module")
def sp(data_home):
    return opensps.StellarPopulation(zcontinuous=1, libraries=LIBS, sfh=4, tau=2.0, dust2=0.3,
                                     add_neb_emission=True, logzsol=-0.2)


def reference(params, zc=1, **extra):
    """The same model through a Context set up from the python-fsps defaults."""
    c = opensps.Context(*LIBS)
    c.set_params(opensps.compat.DEFAULTS)
    c.set_params(params, **extra)
    out = c.compute_csp(zc)
    c.close()
    return out


def test_spectrum_equals_context(sp):
    wave, spec = sp.get_spectrum(tage=3.0)
    ref = reference(dict(sp.params.items()))
    assert_identical(spec, ref["spec"][0])
    assert_identical(wave, sp.context.wavelengths)
    assert sp.stellar_mass == ref["mass"][0] and sp.log_lbol == ref["lbol"][0]
    assert sp.formed_mass == ref["mformed"][0] and sp.sfr == ref["sfr"][0]
    assert sp.dust_mass == ref["mdust"][0] and sp.log_age == ref["age"][0]
    assert_identical(sp.emline_luminosity, ref["emlines"][0])
    assert sp.emline_luminosity.shape == (len(sp.emline_wavelengths),)


def test_peraa(sp):
    wave, fnu = sp.get_spectrum(tage=3.0)
    _, flam = sp.get_spectrum(tage=3.0, peraa=True)
    assert_identical(flam, fnu * (3e18 / wave**2))


def test_all_ages(sp):
    wave, spec = sp.get_spectrum(tage=0.0)
    assert spec.shape == (len(sp.ssp_ages), len(wave))
    assert sp.stellar_mass.shape == (len(sp.ssp_ages),)
    assert_identical(sp.log_age, sp.ssp_ages)
    mags = sp.get_mags(tage=0.0)
    assert mags.shape == (len(sp.ssp_ages), len(opensps.list_filters()))


def test_mags_bands(sp):
    allm = sp.get_mags(tage=2.0)
    sel = sp.get_mags(tage=2.0, bands=["SDSS_r", "v", "galex_FUV"])
    names = opensps.list_filters()
    assert_identical(sel, allm[[names.index(b) for b in ("sdss_r", "v", "galex_fuv")]])
    ref = reference(dict(sp.params.items()))
    assert_identical(allm, ref["mags"][0])
    with pytest.raises(KeyError):
        sp.get_mags(tage=2.0, bands=["no_such_band"])


def test_redshift_override(sp):
    m1 = sp.get_mags(tage=2.0, redshift=1.0, bands=["sdss_r"])
    assert sp.params["zred"] == 0.0
    ref = reference(dict(sp.params.items()), zred=1.0)
    names = opensps.list_filters()
    assert m1[0] == ref["mags"][0][names.index("sdss_r")]


def test_params_semantics(sp):
    assert "imf1" in sp.params.all_params and "dust2" in sp.params.csp_params
    with pytest.raises(KeyError):
        sp.params["no_such_param"] = 1
    sp.get_spectrum(tage=1.0)
    assert not sp.params.dirty
    sp.params["dust2"] = sp.params["dust2"]
    assert not sp.params.dirty
    sp.params["dust2"] = np.array([0.5])  # Prospector passes size-1 arrays
    assert sp.params.dirtiness == 1
    sp.params["imf1"] = 1.4
    assert sp.params.dirtiness == 2
    sp.params["imf1"] = 1.3
    sp.params["dust2"] = 0.3
    with pytest.raises(TypeError):
        opensps.StellarPopulation(not_a_param=1)


def test_tabular_sfh_like_prospector(data_home):
    """FastStepBasis: sfh=3, set_tabular_sfh, spectrum / mtot."""
    sp = opensps.StellarPopulation(zcontinuous=1, libraries=LIBS, logzsol=-0.5)
    sp.params["sfh"] = 3
    # Bins of lookback time 0-0.1, 0.1-1, 1-10 Gyr with 1e8, 1e9, 5e9 M_sun formed, at
    # tage = 10 Gyr: constant SFR per bin, the table in forward time (edges +- 1e-4 Gyr)
    mass = np.array([1e8, 1e9, 5e9])
    tmax, e = 10.0, 1e-4
    t = np.array([0.0, 9.0 - e, 9.0 + e, 9.9 - e, 9.9 + e, tmax])
    rate = np.array([5e9 / 9e9, 1e9 / 0.9e9, 1e8 / 0.1e9])
    sfr = rate[[0, 0, 1, 1, 2, 2]]
    sp.set_tabular_sfh(t, sfr)
    wave, spec = sp.get_spectrum(tage=tmax, peraa=False)
    assert np.all(np.isfinite(spec)) and spec.max() > 0
    assert 0.3 < sp.stellar_mass / mass.sum() < 1.0
    with pytest.raises(ValueError):
        sp.set_tabular_sfh([1.0, 0.5], [1.0, 1.0])


def test_install_as_fsps(data_home):
    saved = sys.modules.get("fsps")
    try:
        opensps.install_as_fsps()
        import fsps

        assert fsps.StellarPopulation is opensps.StellarPopulation
        assert "sdss_r" in fsps.list_filters()
    finally:
        if saved is None:
            sys.modules.pop("fsps", None)
        else:
            sys.modules["fsps"] = saved


def test_independent_instances(data_home):
    a = opensps.StellarPopulation(libraries=LIBS, sfh=1, tau=1.0, zmet=5)
    b = opensps.StellarPopulation(libraries=LIBS, sfh=1, tau=1.0, zmet=12)
    sa = a.get_spectrum(tage=5.0)[1]
    sb = b.get_spectrum(tage=5.0)[1]
    assert not np.array_equal(sa, sb)
    assert_identical(a.get_spectrum(tage=5.0)[1], sa)
