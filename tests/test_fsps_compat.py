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
