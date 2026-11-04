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
            ctx["add_dust_emission"] = False  # bool
            ctx["smooth_velocity"] = True      # bool
            ctx.set_fast_mode(True)
            ctx.set_fast_mode(False)

            with self.assertRaises(opensps.OpenSPSError):
                ctx.set_param("nonexistent_param", 123.45)

    def test_destroyed_context_error(self):
        ctx = opensps.Context(zmet=1, isoc="mist", spec="miles", dust="DL07")
        ctx.destroy()
        with self.assertRaises(RuntimeError):
            _ = ctx.handle
        with self.assertRaises(RuntimeError):
            _ = ctx.nspec

    def test_context_operator_methods(self):
        with opensps.Context(zmet=1, isoc="mist", spec="miles", dust="DL07") as ctx:
            with self.assertRaises(opensps.OpenSPSError):
                ctx.evaluate_ssp_operator()

            ctx.build_ssp_operator()
            self.assertTrue(ctx.has_ssp_operator)

            # Evaluate directly through context method
            mass, lbol, spec = ctx.evaluate_ssp_operator(imf_type=2, imf_params=[1.3, 2.3, 2.3, 0.0, 0.0])
            self.assertEqual(len(mass), ctx.ntfull)

            ctx.free_ssp_operator()
            self.assertFalse(ctx.has_ssp_operator)

    def test_error_class_repr(self):
        err = opensps.OpenSPSError(101, "Test error message")
        self.assertEqual(err.status, 101)
        self.assertIn("Test error message", str(err))

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
