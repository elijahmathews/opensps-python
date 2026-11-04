"""Unit tests for opensps.CSPBatch forward calculations."""

import os
import unittest
import numpy as np
import opensps


class TestBatch(unittest.TestCase):
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

    def test_forward_batch_shapes(self):
        N = 4
        params = {
            "dust2": [0.1, 0.3, 0.6, 1.0],
            "tage": [2.0, 5.0, 8.0, 11.0],
            "tau": 1.5,  # broadcast scalar to N
        }
        batch = self.ctx.batch(params)
        self.assertEqual(batch.n_models, N)

        res = batch.compute(compute_spec=True, compute_emlines=True, compute_indices=True)
        self.assertEqual(res.mags.shape, (N, self.ctx.nbands))
        self.assertEqual(res.spec.shape, (N, self.ctx.nspec))
        self.assertEqual(res.mass.shape, (N,))
        self.assertEqual(res.lbol.shape, (N,))
        self.assertEqual(res.sfr.shape, (N,))
        self.assertEqual(res.mdust.shape, (N,))
        self.assertEqual(res.emlines.shape, (N, self.ctx.nemline))
        self.assertEqual(res.indx.shape, (N, self.ctx.nindx))

        self.assertTrue(np.all(np.isfinite(res.mags)))
        self.assertTrue(np.all(res.mass > 0.0))

    def test_forward_batch_parts(self):
        N = 3
        params = {
            "dust2": [0.2, 0.4, 0.8],
            "tage": 10.0,
        }
        batch = self.ctx.batch(params)
        res = batch.compute(parts=True)

        self.assertIsNotNone(res.young)
        self.assertIsNotNone(res.old)
        self.assertEqual(res.young.shape, (N, self.ctx.nspec))
        self.assertEqual(res.old.shape, (N, self.ctx.nspec))

    def test_forward_batch_with_weights(self):
        N = 3
        nt = self.ctx.ntfull
        # Non-parametric SFH weights
        weights = np.zeros((N, nt), dtype=np.float64)
        for m in range(N):
            weights[m, :] = np.exp(-0.5 * ((np.arange(nt) - (5 + 3 * m)) / 3.0) ** 2)
            weights[m, :] /= np.sum(weights[m, :])

        params = {"dust2": [0.1, 0.5, 1.0]}
        batch = self.ctx.batch(params, ssp_weights=weights)
        res = batch.compute(compute_spec=True)

        self.assertEqual(res.mags.shape, (N, self.ctx.nbands))
        self.assertEqual(res.spec.shape, (N, self.ctx.nspec))
        self.assertTrue(np.all(np.isfinite(res.mags)))


if __name__ == "__main__":
    unittest.main()
