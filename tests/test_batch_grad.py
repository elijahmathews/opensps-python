"""Unit tests verifying OpenSPS analytical gradients against numerical finite differences."""

import os
import unittest
import numpy as np
import opensps


class TestBatchGradients(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data_home = os.environ.get("OPENSPS_DATA_HOME")
        if not data_home:
            here = os.path.dirname(os.path.abspath(__file__))
            default_data = os.path.abspath(os.path.join(here, "../../libopensps/tests/data/opensps-test-data"))
            if os.path.isdir(default_data):
                os.environ["OPENSPS_DATA_HOME"] = default_data

        cls.ctx = opensps.Context(zmet=-1, isoc="mist", spec="miles", dust="DL07")
        cls.ctx["add_dust_emission"] = 0
        cls.ctx["add_neb_emission"] = 0
        cls.ctx["add_agn_dust"] = 1
        cls.ctx["tage"] = 10.0

    @classmethod
    def tearDownClass(cls):
        cls.ctx.destroy()

    def test_gradient_against_finite_differences(self):
        N = 2
        dust2_base = np.array([0.2, 0.6], dtype=np.float64)
        fagn_base = np.array([0.05, 0.15], dtype=np.float64)

        params = {"dust2": dust2_base.copy(), "fagn": fagn_base.copy()}
        batch = self.ctx.batch(params)

        # Loss = sum of mags
        nbands = self.ctx.nbands
        mags_bar = np.ones((N, nbands), dtype=np.float64)

        grad_res = batch.compute_gradients(mags_bar=mags_bar)

        adj_dust2 = grad_res.params_bar["dust2"]
        adj_fagn = grad_res.params_bar["fagn"]

        # Finite differences check
        h = 1e-5
        for m in range(N):
            # Check dust2
            p_plus = {"dust2": dust2_base.copy(), "fagn": fagn_base.copy()}
            p_plus["dust2"][m] += h
            res_plus = self.ctx.batch(p_plus).compute()

            p_minus = {"dust2": dust2_base.copy(), "fagn": fagn_base.copy()}
            p_minus["dust2"][m] -= h
            res_minus = self.ctx.batch(p_minus).compute()

            fd_dust2 = (np.sum(res_plus.mags[m]) - np.sum(res_minus.mags[m])) / (2.0 * h)
            rel_err_dust2 = abs(adj_dust2[m] - fd_dust2) / max(abs(fd_dust2), 1e-4)
            self.assertLess(rel_err_dust2, 1e-4, f"dust2 gradient mismatch for model {m}")

            # Check fagn
            p_plus = {"dust2": dust2_base.copy(), "fagn": fagn_base.copy()}
            p_plus["fagn"][m] += h
            res_plus = self.ctx.batch(p_plus).compute()

            p_minus = {"dust2": dust2_base.copy(), "fagn": fagn_base.copy()}
            p_minus["fagn"][m] -= h
            res_minus = self.ctx.batch(p_minus).compute()

            fd_fagn = (np.sum(res_plus.mags[m]) - np.sum(res_minus.mags[m])) / (2.0 * h)
            rel_err_fagn = abs(adj_fagn[m] - fd_fagn) / max(abs(fd_fagn), 1e-4)
            self.assertLess(rel_err_fagn, 1e-4, f"fagn gradient mismatch for model {m}")

    def test_gradient_with_weights_z_and_cotangents(self):
        N = 2
        nt = self.ctx.ntfull
        nz = self.ctx.nz
        wz = np.zeros((N, nt, nz), dtype=np.float64)
        wz[:, 0, 0] = 1.0

        batch = self.ctx.batch({"dust2": [0.3, 0.7]}, ssp_weights_z=wz)
        grad_res = batch.compute_gradients(
            mags_bar=np.ones((N, self.ctx.nbands)),
            mass_bar=np.ones(N),
            lbol_bar=np.ones(N),
        )

        self.assertIsNotNone(grad_res.weights_z_bar)
        self.assertEqual(grad_res.weights_z_bar.shape, (N, nt, nz))
        self.assertIn("dust2", grad_res.params_bar)

    def test_cotangent_validation_errors(self):
        batch = self.ctx.batch({"dust2": [0.2, 0.4]})
        # Wrong mags_bar shape
        with self.assertRaises(ValueError):
            batch.compute_gradients(mags_bar=np.ones((3, self.ctx.nbands)))
        # Wrong mass_bar shape
        with self.assertRaises(ValueError):
            batch.compute_gradients(mass_bar=np.ones(3))


if __name__ == "__main__":
    unittest.main()
