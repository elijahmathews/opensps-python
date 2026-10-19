"""The batch entry point: equal to direct C calls, to batches of one, to single CSPs."""

import numpy as np
import pytest

import opensps
from conftest import assert_identical

N = 8
RNG = np.random.default_rng(42)
PARAMS = dict(
    logzsol=RNG.uniform(-1.5, 0.15, N), dust2=RNG.uniform(0.0, 1.0, N),
    dust_index=RNG.uniform(-1.0, 0.2, N), tau=RNG.uniform(0.3, 8.0, N),
    tage=RNG.uniform(1.0, 12.0, N), fagn=RNG.uniform(0.0, 0.3, N), zred=RNG.uniform(0.0, 2.0, N),
    gas_logu=RNG.uniform(-3.5, -1.5, N), duste_umin=RNG.uniform(0.5, 20.0, N),
)
SETTINGS = dict(sfh=4, dust_type=4, add_neb_emission=1, add_igm_absorption=1, add_agn_dust=1)


@pytest.fixture
def bctx(fresh):
    fresh.set_params(SETTINGS)
    return fresh


def test_batch_equals_c(bctx, cref):
    out = bctx.compute_csp_batch(PARAMS, emlines=True)
    h = cref.create()
    try:
        for k, v in SETTINGS.items():
            cref.set(h, k, v, True)
        ref = cref.batch(h, N, PARAMS)
    finally:
        cref.destroy(h)
    for k in ("mags", "spec", "mass", "lbol", "mdust", "sfr", "emlines"):
        assert_identical(out[k], ref[k])
    assert out["spec"].shape == (N, bctx.nspec) and out["mags"].shape == (N, bctx.nbands)


def test_batch_equals_one_at_a_time(bctx):
    out = bctx.compute_csp_batch(PARAMS, emlines=True)
    for m in range(N):
        one = bctx.compute_csp_batch({k: v[m:m + 1] for k, v in PARAMS.items()}, emlines=True)
        for k in ("mags", "spec", "mass", "lbol", "emlines"):
            np.testing.assert_allclose(one[k][0], out[k][m], rtol=1e-12, atol=0)


def test_batch_equals_single_csp(bctx):
    """Batch against the reference single-model path (compute_csp_scenario, zcontinuous=1);
    tolerance of the core's batch test (tests/test_csp_batch.f90)."""
    out = bctx.compute_csp_batch(PARAMS, emlines=True)
    for m in range(N):
        bctx.set_params({k: v[m] for k, v in PARAMS.items()})
        one = bctx.compute_csp(1)
        np.testing.assert_allclose(out["spec"][m], one["spec"][0], rtol=1e-10, atol=1e-30)
        bright = one["mags"][0] < np.nanmin(one["mags"][0]) + 30
        np.testing.assert_allclose(out["mags"][m][bright], one["mags"][0][bright], atol=1e-9, rtol=0)
        np.testing.assert_allclose(out["mass"][m], one["mass"][0], rtol=1e-12)
        np.testing.assert_allclose(out["lbol"][m], one["lbol"][0], atol=1e-12)


def test_ssp_weights(bctx, cref):
    nt = bctx.ntfull
    w = RNG.uniform(0, 1, (N, nt))
    w /= w.sum(axis=1, keepdims=True)
    p = dict(dust2=PARAMS["dust2"], zred=PARAMS["zred"])
    out = bctx.compute_csp_batch(p, ssp_weights=w)
    h = cref.create()
    try:
        for k, v in SETTINGS.items():
            cref.set(h, k, v, True)
        ref = cref.batch(h, N, p, weights=w)
    finally:
        cref.destroy(h)
    assert_identical(out["spec"], ref["spec"])
    assert_identical(out["mags"], ref["mags"])
    with pytest.raises(ValueError):
        bctx.compute_csp_batch(p, ssp_weights=w[:, :-1])


def test_out_buffers_zero_copy(bctx):
    first = bctx.compute_csp_batch(PARAMS)
    bufs = {k: np.empty_like(v) for k, v in first.items()}
    res = bctx.compute_csp_batch(PARAMS, out=bufs)
    for k in bufs:
        assert res[k] is bufs[k]
        assert_identical(res[k], first[k])
    with pytest.raises(ValueError):
        bctx.compute_csp_batch(PARAMS, out={"spec": np.empty((N, 3))})


def test_kernel_precision_spectra(bctx):
    out = bctx.compute_csp_batch(PARAMS)
    host = bctx.batch_spectra_host()
    assert host.shape == out["spec"].shape and not host.flags.writeable
    np.testing.assert_array_equal(host.astype(np.float64), out["spec"])
    dev = bctx.batch_spectra_device()
    assert dev.shape == out["spec"].shape and dev.ptr != 0
    assert dev.__cuda_array_interface__["shape"] == out["spec"].shape


def test_scalar_broadcast_and_errors(bctx):
    out = bctx.compute_csp_batch(dict(dust2=PARAMS["dust2"], tage=5.0))
    assert out["mags"].shape[0] == N
    with pytest.raises(opensps.FSPSError) as e:
        bctx.compute_csp_batch(dict(imf1=np.full(N, 1.3)))
    assert e.value.status == 2
    with pytest.raises(opensps.FSPSError) as e:
        bctx.compute_csp_batch(dict(not_a_parameter=np.zeros(N)))
    assert e.value.status == 3
    with pytest.raises(ValueError):
        bctx.compute_csp_batch(dict(dust2=np.zeros(N), tau=np.zeros(N + 1)))
    with pytest.raises(ValueError):
        bctx.compute_csp_batch({})
