"""Unit tests for opensps.Context."""

import os
import unittest
import numpy as np
import opensps


class TestContext(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data_home = os.environ.get("OPENSPS_DATA_HOME")
        if not data_home:
            # Fall back to repo test data
            here = os.path.dirname(os.path.abspath(__file__))
            default_data = os.path.abspath(os.path.join(here, "../../libopensps/tests/data/opensps-test-data"))
            if os.path.isdir(default_data):
                os.environ["OPENSPS_DATA_HOME"] = default_data

    def test_version_and_metadata(self):
        major, minor, patch = opensps.get_driver_version()
        self.assertEqual(major, 2)
        self.assertEqual(minor, 1)
        self.assertEqual(opensps.get_nbgrad(), 33)

        name_fagn = opensps.get_grad_param_name(opensps.BGRAD.FAGN)
        self.assertEqual(name_fagn, "fagn")
        name_dust2 = opensps.get_grad_param_name(opensps.BGRAD.DUST2)
        self.assertEqual(name_dust2, "dust2")

    def test_context_dimensions_and_grids(self):
        with opensps.Context(zmet=-1, isoc="mist", spec="miles", dust="DL07") as ctx:
            self.assertGreater(ctx.nspec, 0)
            self.assertGreater(ctx.ntfull, 0)
            self.assertGreater(ctx.nbands, 0)
            self.assertGreater(ctx.nz, 0)
            self.assertGreater(ctx.nemline, 0)
            self.assertGreater(ctx.zsol, 0.0)

            # Check grids
            wave = ctx.wavelengths
            self.assertEqual(len(wave), ctx.nspec)
            self.assertTrue(np.all(np.diff(wave) > 0))

            ages = ctx.log_ages
            self.assertEqual(len(ages), ctx.ntfull)

            zlegend = ctx.zlegend
            self.assertEqual(len(zlegend), ctx.nz)

            emlambda = ctx.emlambda
            self.assertEqual(len(emlambda), ctx.nemline)

    def test_parameter_setting(self):
        with opensps.Context(zmet=1, isoc="mist", spec="miles", dust="DL07") as ctx:
            ctx.set_param("dust2", 0.5)
            ctx["tage"] = 10.0
            ctx["add_neb_emission"] = 1
            ctx["imf_type"] = 2
            ctx.set_fast_mode(True)
            ctx.set_fast_mode(False)

    def test_compute_ssp(self):
        with opensps.Context(zmet=1, isoc="mist", spec="miles", dust="DL07") as ctx:
            spec, mass, lbol = ctx.compute_ssp()
            self.assertEqual(spec.shape, (ctx.ntfull, ctx.nspec))
            self.assertEqual(mass.shape, (ctx.ntfull,))
            self.assertEqual(lbol.shape, (ctx.ntfull,))
            self.assertTrue(np.all(mass > 0.0))
            self.assertTrue(np.all(np.isfinite(lbol)))


if __name__ == "__main__":
    unittest.main()
