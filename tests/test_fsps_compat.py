"""Unit tests for opensps.StellarPopulation (python-fsps compatibility layer)."""

import os
import unittest
import numpy as np
import opensps


class TestFSPSCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data_home = os.environ.get("OPENSPS_DATA_HOME")
        if not data_home:
            here = os.path.dirname(os.path.abspath(__file__))
            default_data = os.path.abspath(os.path.join(here, "../../libopensps/tests/data/opensps-test-data"))
            if os.path.isdir(default_data):
                os.environ["OPENSPS_DATA_HOME"] = default_data

    def test_stellar_population_api(self):
        sp = opensps.StellarPopulation(compute_vega_mags=False, zcontinuous=1)

        # Check parameter dict
        sp.params["logzsol"] = -0.5
        sp.params["dust2"] = 0.3
        self.assertEqual(sp.params["logzsol"], -0.5)
        self.assertEqual(sp.params["dust2"], 0.3)
        self.assertTrue("dust2" in sp.params)
        self.assertIn("dust2", list(sp.params.keys()))
        self.assertIn(0.3, list(sp.params.values()))
        self.assertIn(("dust2", 0.3), list(sp.params.items()))
        self.assertIn("dust2", repr(sp.params))

        # Check properties
        self.assertGreater(len(sp.wavelengths), 0)
        self.assertGreater(len(sp.ssp_ages), 0)
        self.assertGreater(len(sp.zlegend), 0)
        self.assertGreater(len(sp.emline_wavelengths), 0)
        self.assertEqual(len(sp.libraries), 3)
        self.assertGreater(sp.solar_metallicity, 0.0)

        # Check get_spectrum
        wave, spec = sp.get_spectrum(tage=10.0, peraa=False)
        self.assertEqual(len(wave), len(sp.wavelengths))
        self.assertEqual(len(spec), len(sp.wavelengths))
        self.assertTrue(np.all(np.isfinite(spec)))
        self.assertGreater(sp.stellar_mass, 0.0)
        self.assertTrue(np.isfinite(sp.dust_mass))
        self.assertEqual(len(sp.emline_luminosity), sp.context.nemline)

        # Check get_spectrum with peraa
        wave, spec_aa = sp.get_spectrum(tage=10.0, peraa=True)
        self.assertEqual(len(spec_aa), len(sp.wavelengths))

        # Check get_mags
        mags = sp.get_mags(tage=10.0)
        self.assertEqual(len(mags), sp.context.nbands)
        self.assertTrue(np.all(np.isfinite(mags)))

        # Check get_mags with bands filter
        mags_sub = sp.get_mags(tage=10.0, bands=[0, 1, 2])
        self.assertEqual(len(mags_sub), 3)


if __name__ == "__main__":
    unittest.main()


class TestPythonFSPSSemantics(unittest.TestCase):
    """What code written for python-fsps (Prospector's CSPSpecBasis) relies on."""

    @classmethod
    def setUpClass(cls):
        TestFSPSCompatibility.setUpClass()
        cls.sp = opensps.StellarPopulation(zcontinuous=1)
        for k, v in dict(sfh=4, imf_type=2, dust_type=0, tau=1.0, dust2=0.3, tage=3.0).items():
            cls.sp.params[k] = v

    def test_all_params_lists_fsps_names(self):
        for name in ("logzsol", "dust2", "tage", "tau", "sfh", "imf_type", "zred"):
            self.assertIn(name, self.sp.params.all_params)

    def test_logzsol_reaches_the_spectrum(self):
        """A batch not given logzsol uses one SSP metallicity: the object must pass its own."""
        sp = self.sp
        sp.params["logzsol"] = 0.0
        _, hi = sp.get_spectrum(tage=3.0)
        mass_hi = sp.stellar_mass
        sp.params["logzsol"] = -1.0
        _, lo = sp.get_spectrum(tage=3.0)
        self.assertGreater(np.max(np.abs(hi / np.maximum(lo, 1e-300) - 1.0)[lo > 1e-20]), 0.05)
        self.assertNotEqual(mass_hi, sp.stellar_mass)
        # and it equals the same model given as per-model batch parameters
        ref = sp.context.batch({"tage": 3.0, "tau": 1.0, "dust2": 0.3, "logzsol": -1.0}).compute(compute_spec=True)
        np.testing.assert_allclose(lo, ref.spec[0], rtol=1e-12)
        self.assertEqual(sp.stellar_mass, float(ref.mass[0]))

    def test_unchanged_set_is_not_pushed(self):
        """Re-setting a parameter to its value must not reach the context (it can invalidate the SSP basis)."""
        calls = []
        orig = self.sp.context.set_param
        self.sp.context.set_param = lambda k, v: (calls.append(k), orig(k, v))[1]
        try:
            self.sp.params["imf_type"] = 2
            self.sp.params["tau"] = 1.0
            self.assertEqual(calls, [])
            self.sp.params["imf_type"] = 1
            self.assertEqual(calls, ["imf_type"])
            self.sp.params["imf_type"] = 2
        finally:
            self.sp.context.set_param = orig

    def test_numpy_scalars_are_accepted(self):
        self.sp.params["sfh"] = np.int64(4)
        self.sp.params["dust2"] = np.float64(0.3)

    def test_pickle_resolves_to_the_same_object_in_this_process(self):
        import pickle
        self.assertIs(pickle.loads(pickle.dumps(self.sp)), self.sp)
