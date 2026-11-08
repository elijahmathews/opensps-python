"""Tests of the SSP weights of piecewise-linear SFHs."""

import os
import unittest
import numpy as np
import opensps
from opensps import sfh


class TestSFHWeights(unittest.TestCase):
    def test_weights_conserve_mass_and_are_positive(self):
        log_ages = 5.0 + 0.05 * np.arange(107)
        edges = np.array([[0.0, 3e7, 1e8, 1e9, 5e9]])
        rate = np.array([[2.0, 0.5, 1.0, 0.1]])
        t, s = sfh.step_table(edges, rate)
        w = sfh.ssp_weights_from_table(log_ages, t, s)
        formed = (rate * np.diff(edges, axis=1)).sum()
        self.assertTrue(np.all(w >= 0))
        self.assertAlmostEqual(w.sum() / formed, 1.0, places=6)
        # nothing is older than the oldest bin edge
        self.assertEqual(w[0, 10.0 ** log_ages > 5.1e9 * 1.2].max(), 0.0)

    def test_constant_sfr_weights_are_the_hat_integrals(self):
        log_ages = np.log10([1e6, 1e7, 1e8, 1e9])
        t = np.array([[0.0, 1e9]])
        s = np.array([[1.0, 1.0]])
        w = sfh.ssp_weights_from_table(log_ages, t, s)[0]
        a = 10.0 ** log_ages
        # interior node: hat linear in log age between its neighbours, integrated against a constant
        x = log_ages
        def hat_integral(i):
            r = 0.0
            if i > 0:    # rising on [a_{i-1}, a_i]: int (log tau - x_{i-1}) / dx dtau
                lo, hi = a[i - 1], a[i]
                f = lambda tau: (tau * np.log10(tau) - tau / np.log(10.0)) - x[i - 1] * tau
                r += (f(hi) - f(lo)) / (x[i] - x[i - 1])
            if i < len(a) - 1:
                lo, hi = a[i], a[i + 1]
                f = lambda tau: x[i + 1] * tau - (tau * np.log10(tau) - tau / np.log(10.0))
                r += (f(hi) - f(lo)) / (x[i + 1] - x[i])
            return r
        for i in (1, 2):
            self.assertAlmostEqual(w[i] / hat_integral(i), 1.0, places=9)
        self.assertAlmostEqual(w.sum(), 1e9, delta=1e3)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            sfh.ssp_weights_from_table(np.arange(5.0, 6.0, 0.1), [[0.0, 1.0, 0.5]], [[1.0, 1.0, 1.0]])
        with self.assertRaises(ValueError):
            sfh.step_table(np.zeros((1, 3)), np.ones((1, 3)))

    def test_matches_the_context_tabular_path(self):
        with opensps.Context(zmet=-1, isoc="mist", spec="miles", dust="DL07") as ctx:
            edges = np.array([[0.0, 3e7, 1e8, 1e9, 4e9]])
            rate = np.array([[3.0, 0.2, 1.0, 0.4]])
            t, s = sfh.step_table(edges, rate)
            w = sfh.ssp_weights_from_table(ctx.log_ages, t, s)
            formed = float(w.sum())
            by_weights = ctx.batch({"dust2": 0.3}, ssp_weights=w / formed).compute(compute_spec=False)

            # the same table through the context: forward time since the first star formed, in Gyr
            ctx.set_param("sfh", 3)
            ctx.set_param("tage", float(t[0, -1]) / 1e9)
            ctx.set_sfh_tab((t[0, -1] - t[0])[::-1] / 1e9, s[0][::-1])
            by_table = ctx.batch({"dust2": 0.3}).compute(compute_spec=False)
            np.testing.assert_allclose(by_weights.mags[0, :40] + 2.5 * np.log10(1.0 / formed),
                                       by_table.mags[0, :40], atol=1e-5)
            np.testing.assert_allclose(by_weights.mass[0] * formed, by_table.mass[0], rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
