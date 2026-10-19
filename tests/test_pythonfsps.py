"""Optional: against python-fsps, if it is installed (it is not a dependency).

python-fsps wraps an unpatched legacy FSPS, whose constant gsig4pi = 1/4.13e10 makes every
main-library spectrum 1.77% brighter than the exact 1/(4 pi sigma) used by libopensps (see the
core's tools/legacy_parity/README.md). With identical data the expected agreement for an SSP
(Chabrier, solar metallicity, MIST + MILES, no circumstellar AGB dust) at log t >= 7.2 is:

- spectra: opensps / python-fsps = 0.9827 +- 0.001 (median over 3000-9000 A per age);
- surviving mass and L_bol: 1e-3 dex.

python-fsps reads its data from its own SPS_HOME (set it to a legacy data tree that matches
this package's data); its libraries must be MIST + MILES.
"""

import importlib

import numpy as np
import pytest

import opensps
from conftest import LIBS


def _python_fsps():
    try:
        mod = importlib.import_module("fsps")
    except Exception:
        return None
    if getattr(mod, "__opensps__", False):
        return None
    return mod


fsps = _python_fsps()
pytestmark = pytest.mark.skipif(fsps is None, reason="python-fsps not installed")


def test_ssp_against_python_fsps():
    ref = fsps.StellarPopulation(zcontinuous=0)
    libs = [x.decode() if isinstance(x, bytes) else x for x in ref.libraries]
    if libs[0].lower() != "mist" or libs[1].lower() != "miles":
        pytest.skip(f"python-fsps built with {libs}, not MIST + MILES")
    zsol = ref.solar_metallicity
    zmet = int(np.argmin(np.abs(ref.zlegend - zsol))) + 1
    ref.params["zmet"] = zmet
    ref.params["add_agb_dust_model"] = False
    w_ref, s_ref = ref.get_spectrum(tage=0.0)
    m_ref, l_ref, age_ref = ref.stellar_mass, ref.log_lbol, ref.log_age

    sp = opensps.StellarPopulation(zcontinuous=0, libraries=LIBS, zmet=zmet,
                                   add_agb_dust_model=False)
    w, s = sp.get_spectrum(tage=0.0)
    assert np.allclose(sp.zlegend, ref.zlegend, rtol=1e-6)
    np.testing.assert_allclose(w, w_ref, rtol=1e-10)
    np.testing.assert_allclose(sp.log_age, age_ref, atol=1e-6)

    old = sp.log_age >= 7.2
    np.testing.assert_allclose(np.log10(sp.stellar_mass[old]), np.log10(m_ref[old]), atol=1e-3)
    np.testing.assert_allclose(sp.log_lbol[old], l_ref[old], atol=1e-3)
    band = (w > 3000) & (w < 9000)
    ratio = np.median(s[old][:, band] / s_ref[old][:, band], axis=1)
    print("spectrum ratio opensps / python-fsps: min %.5f max %.5f" % (ratio.min(), ratio.max()))
    assert np.all(np.abs(ratio - 0.9827) < 1e-3)
