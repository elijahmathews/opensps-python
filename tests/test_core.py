"""The Context wrapper against direct C calls (bit for bit), grids, errors."""

import numpy as np
import pytest

import opensps
from conftest import assert_identical

# A full-physics single model: tau SFH, Calzetti dust with dust emission, nebular emission,
# AGN, IGM at z = 0.5
MODEL_INT = dict(sfh=4, dust_type=2, add_neb_emission=1, add_igm_absorption=1, add_agn_dust=1)
MODEL_FLOAT = dict(tau=1.5, tage=4.0, logzsol=-0.35, dust2=0.4, dust1=0.2, fagn=0.05,
                   agn_tau=20.0, gas_logu=-2.5, zred=0.5, duste_umin=2.0)


def test_api_version():
    assert opensps.api_version() == opensps.header_api_version()
    assert opensps.api_version() >= (1, 1, 0)


def test_grids_equal_c(ctx, cref):
    h = cref.create()
    try:
        assert ctx.nspec == cref.dim(h, "nspec") and ctx.ntfull == cref.dim(h, "ntfull")
        assert_identical(ctx.wavelengths, cref.grid(h, "lambda", ctx.nspec))
        assert_identical(ctx.log_ages, cref.grid(h, "ages", ctx.ntfull))
        assert_identical(ctx.zlegend, cref.grid(h, "zlegend", ctx.nz))
        assert_identical(ctx.emline_wavelengths, cref.grid(h, "emlambda", ctx.nemline))
    finally:
        cref.destroy(h)
    assert not ctx.wavelengths.flags.writeable
    weff, mvega, msun = ctx.filter_data()
    assert weff.shape == (ctx.nbands,) and mvega[0] == 0.0
    assert len(opensps.list_filters()) == ctx.nbands
    assert len(opensps.emline_names()) == ctx.nemline


def test_ssp_equals_c(fresh, cref):
    fresh.set_params(zmet=7, imf_type=1)
    spec, mass, lbol = fresh.compute_ssp()
    h = cref.create()
    try:
        cref.set(h, "zmet", 7, True)
        cref.set(h, "imf_type", 1, True)
        rspec, rmass, rlbol = cref.compute_ssp(h)
    finally:
        cref.destroy(h)
    assert spec.shape == (fresh.ntfull, fresh.nspec)
    assert_identical(spec, rspec)
    assert_identical(mass, rmass)
    assert_identical(lbol, rlbol)


@pytest.mark.parametrize("zc", [0, 1])
def test_csp_equals_c(fresh, cref, zc):
    fresh.set_params(MODEL_INT, zmet=9, **MODEL_FLOAT)
    out = fresh.compute_csp(zc)
    h = cref.create()
    try:
        cref.set(h, "zmet", 9, True)
        for k, v in MODEL_INT.items():
            cref.set(h, k, v, True)
        for k, v in MODEL_FLOAT.items():
            cref.set(h, k, v, False)
        ref = cref.compute_csp(h, zc)
    finally:
        cref.destroy(h)
    assert set(out) == set(ref)
    for k in ref:
        assert_identical(out[k], ref[k])
    assert out["spec"].shape == (1, fresh.nspec)


def test_csp_all_ages(fresh):
    fresh.set_params(sfh=1, tau=2.0, tage=0.0)
    out = fresh.compute_csp(0, outputs=["age", "spec", "mass"])
    assert set(out) == {"age", "spec", "mass"}
    assert out["spec"].shape == (fresh.ntfull, fresh.nspec)
    assert_identical(out["age"], fresh.log_ages)


def test_tabular_sfh(fresh):
    fresh.set_params(sfh=3, tage=5.0)
    with pytest.raises(opensps.FSPSError) as e:
        fresh.compute_csp(1)
    assert e.value.status == 311
    fresh.set_tabular_sfh([0.0, 6.0, 13.8], [1.0, 1.0, 1.0])
    tab = fresh.compute_csp(1)
    fresh.set_params(sfh=1, const=1.0)
    const = fresh.compute_csp(1)
    # SFR in M_sun/yr, not normalized: 5e9 M_sun formed by 5 Gyr
    np.testing.assert_allclose(tab["spec"] / 5e9, const["spec"], rtol=1e-8)
    np.testing.assert_allclose(tab["mass"] / 5e9, const["mass"], rtol=1e-8)


def test_errors(fresh):
    with pytest.raises(opensps.UnknownParameterError):
        fresh.set("not_a_parameter", 1.0)
    with pytest.raises(KeyError):
        fresh.set("not_a_parameter", 1.0)
    with pytest.raises(ValueError):
        fresh.set("sfh", 1.5)
    with pytest.raises(opensps.FSPSError) as e:
        fresh.compute_csp(7)
    assert e.value.status == 306
    fresh.set("zmet", 99)
    with pytest.raises(opensps.FSPSError) as e:
        fresh.compute_csp(0)
    assert e.value.status == 304
    c = opensps.Context(setup=False)
    with pytest.raises(opensps.FSPSError):
        c.compute_csp(0)
    c.close()
    with pytest.raises(opensps.FSPSError):
        c.set("tau", 1.0)


def test_fast_mode(fresh):
    fresh.set_params(sfh=1, tage=3.0)
    fresh.set_fast_mode(True)
    out = fresh.compute_csp(0)
    assert "mags" not in out and "spec" in out
    with pytest.raises(opensps.FSPSError) as e:
        fresh.compute_csp(0, outputs=["mags"])
    assert e.value.status == 309


def test_size1_arrays_accepted(fresh):
    fresh.set("dust2", np.array([0.3]))
    fresh.set("sfh", np.array([4.0]))
    with pytest.raises(ValueError):
        fresh.set("dust2", np.array([0.3, 0.4]))


def test_data_home_errors(tmp_path):
    with pytest.raises(opensps.DataNotFoundError):
        opensps.set_data_home(tmp_path)
