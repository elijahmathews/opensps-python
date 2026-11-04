"""Unit tests for opensps.SSPOperator and IMF-free batch evaluation."""

import os
import unittest
import numpy as np
import opensps


class TestSSPOperator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data_home = os.environ.get("OPENSPS_DATA_HOME")
        if not data_home:
            here = os.path.dirname(os.path.abspath(__file__))
            default_data = os.path.abspath(os.path.join(here, "../../libopensps/tests/data/opensps-test-data"))
            if os.path.isdir(default_data):
                os.environ["OPENSPS_DATA_HOME"] = default_data

        cls.ctx = opensps.Context(zmet=-1, isoc="mist", spec="miles", dust="DL07")

    @classmethod
    def tearDownClass(cls):
        cls.ctx.destroy()

    def test_ssp_operator_lifecycle_and_evaluation(self):
        self.assertFalse(self.ctx.has_ssp_operator)

        op = self.ctx.build_ssp_operator(all_metallicities=False)
        self.assertTrue(self.ctx.has_ssp_operator)

        # Evaluate Kroupa
        mass, lbol, spec = op.evaluate(imf_type=2, imf_params=[1.3, 2.3, 2.3, 0.0, 0.0])
        self.assertEqual(mass.shape, (self.ctx.ntfull,))
        self.assertEqual(lbol.shape, (self.ctx.ntfull,))
        self.assertEqual(spec.shape, (self.ctx.ntfull, self.ctx.nspec))
        self.assertTrue(np.all(mass > 0.0))
        self.assertTrue(np.all(np.isfinite(lbol)))

        # Evaluate with defaults (imf_type=None, imf_params=None)
        m_def, l_def, s_def = op.evaluate()
        self.assertEqual(m_def.shape, (self.ctx.ntfull,))

        # Evaluate with short imf_params (< 5 elements, padded internally)
        m_pad, l_pad, s_pad = op.evaluate(imf_type=2, imf_params=[1.2, 2.2])
        self.assertEqual(m_pad.shape, (self.ctx.ntfull,))

        op.free()
        self.assertFalse(self.ctx.has_ssp_operator)

    def test_batch_with_free_imf_parameters(self):
        self.ctx["imf_type"] = 2  # Kroupa
        self.ctx["tage"] = 10.0
        self.ctx.build_ssp_operator(all_metallicities=False)

        N = 3
        params = {
            "dust2": [0.2, 0.4, 0.6],
            "imf1": [1.1, 1.3, 1.5],
            "imf2": [2.1, 2.3, 2.5],
        }

        batch = self.ctx.batch(params)
        res = batch.compute(compute_spec=True)

        self.assertEqual(res.mags.shape, (N, self.ctx.nbands))
        self.assertEqual(res.spec.shape, (N, self.ctx.nspec))
        self.assertTrue(np.all(np.isfinite(res.mags)))

        # Also verify gradients with ssp_ops
        mags_bar = np.ones((N, self.ctx.nbands), dtype=np.float64)
        grad_res = batch.compute_gradients(mags_bar=mags_bar)

        self.assertIn("imf1", grad_res.params_bar)
        self.assertIn("imf2", grad_res.params_bar)
        self.assertTrue(np.all(np.isfinite(grad_res.params_bar["imf1"])))
        self.assertTrue(np.all(np.isfinite(grad_res.params_bar["imf2"])))

        self.ctx.free_ssp_operator()


if __name__ == "__main__":
    unittest.main()
