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

    def test_set_mag_compute(self):
        """Selected bands give the magnitudes of the full set, the others NaN; None restores all."""
        params = {"dust2": [0.1, 0.3, 0.6], "tage": [2.0, 5.0, 8.0], "tau": 1.5}
        full = self.ctx.batch(params).compute(compute_spec=False).mags
        chosen = [0, 3, self.ctx.nbands - 1]
        try:
            self.ctx.set_mag_compute(chosen)
            part = self.ctx.batch(params).compute(compute_spec=False).mags
        finally:
            self.ctx.set_mag_compute(None)
        mask = np.zeros(self.ctx.nbands, dtype=bool)
        mask[chosen] = True
        np.testing.assert_array_equal(part[:, mask], full[:, mask])
        self.assertTrue(np.all(np.isnan(part[:, ~mask])))
        again = self.ctx.batch(params).compute(compute_spec=False).mags
        np.testing.assert_array_equal(again, full)
        with self.assertRaises(ValueError):
            self.ctx.set_mag_compute([self.ctx.nbands])
        with self.assertRaises(ValueError):
            self.ctx.set_mag_compute([-1])

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

    def test_forward_batch_with_weights_z_and_zred(self):
        N = 2
        nt = self.ctx.ntfull
        nz = self.ctx.nz

        # Test both (N, nt, nz) and (nt, nz, N) shapes
        wz_1 = np.zeros((N, nt, nz), dtype=np.float64)
        wz_1[:, 0, 0] = 1.0

        batch_1 = self.ctx.batch({"dust2": [0.2, 0.4]}, ssp_weights_z=wz_1, zred_mags=[0.1, 0.2])
        res_1 = batch_1.compute(compute_spec=False)
        self.assertEqual(res_1.mags.shape, (N, self.ctx.nbands))
        self.assertIsNone(res_1.spec)

        wz_2 = np.zeros((nt, nz, N), dtype=np.float64)
        wz_2[0, 0, :] = 1.0
        batch_2 = self.ctx.batch({"dust2": [0.2, 0.4]}, ssp_weights_z=wz_2)
        res_2 = batch_2.compute(compute_spec=False)
        self.assertEqual(res_2.mags.shape, (N, self.ctx.nbands))

    def test_batch_validation_errors(self):
        # Mismatched parameter lengths
        with self.assertRaises(ValueError):
            self.ctx.batch({"dust2": [0.1, 0.2], "tage": [1.0, 2.0, 3.0]})

        # Invalid weights shape
        with self.assertRaises(ValueError):
            self.ctx.batch({"dust2": [0.1, 0.2]}, ssp_weights=np.zeros((3, self.ctx.ntfull)))

        # Invalid weights_z shape
        with self.assertRaises(ValueError):
            self.ctx.batch({"dust2": [0.1, 0.2]}, ssp_weights_z=np.zeros((2, 5)))

    def test_length_one_and_zero_dim_parameters(self):
        # Regression: float() on a length-1 (non-0-d) array raises TypeError in NumPy >= 2, so a
        # one-model batch given as [x], np.array([x]) or np.array([[x]]) used to fail.
        ref = self.ctx.batch({"dust2": 0.3, "tage": 5.0}).compute()
        for dust2, tage in [
            ([0.3], [5.0]),
            ((0.3,), (5.0,)),
            (np.array([0.3]), np.array([5.0])),
            (np.array(0.3), np.array(5.0)),
            (np.array([[0.3]]), np.array([[5.0]])),
        ]:
            batch = self.ctx.batch({"dust2": dust2, "tage": tage})
            self.assertEqual(batch.n_models, 1)
            res = batch.compute()
            self.assertEqual(res.mags.shape, (1, self.ctx.nbands))
            np.testing.assert_array_equal(res.mags, ref.mags)

    def test_zero_dim_array_alongside_vector(self):
        # Regression: len() of a 0-d ndarray raised in batch-size detection
        batch = self.ctx.batch({"dust2": np.array(0.3), "tage": [2.0, 5.0, 8.0]})
        self.assertEqual(batch.n_models, 3)
        ref = self.ctx.batch({"dust2": 0.3, "tage": [2.0, 5.0, 8.0]}).compute()
        np.testing.assert_array_equal(batch.compute().mags, ref.mags)

    def test_length_one_parameter_broadcasts_over_batch(self):
        res = self.ctx.batch({"tage": [2.0, 5.0, 8.0], "dust2": [0.3]}).compute()
        ref = self.ctx.batch({"tage": [2.0, 5.0, 8.0], "dust2": 0.3}).compute()
        np.testing.assert_array_equal(res.mags, ref.mags)


if __name__ == "__main__":
    unittest.main()
