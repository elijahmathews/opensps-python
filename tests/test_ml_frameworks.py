"""Unit tests for opensps.torch and opensps.jax ML integrations."""

import os
import unittest
from unittest.mock import patch
import numpy as np
import opensps
import opensps.torch
import opensps.jax


class TestMLFrameworks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data_home = os.environ.get("OPENSPS_DATA_HOME")
        if not data_home:
            here = os.path.dirname(os.path.abspath(__file__))
            default_data = os.path.abspath(os.path.join(here, "../../libopensps/tests/data/opensps-test-data"))
            if os.path.isdir(default_data):
                os.environ["OPENSPS_DATA_HOME"] = default_data

        cls.ctx = opensps.Context(zmet=-1, isoc="mist", spec="miles", dust="DL07")
        cls.ctx["tage"] = 10.0
        cls.ctx["add_dust_emission"] = 0
        cls.ctx["add_neb_emission"] = 0
        cls.ctx["add_agn_dust"] = 1

    @classmethod
    def tearDownClass(cls):
        cls.ctx.destroy()

    def test_missing_dependencies_raise_error(self):
        # Verify fallback error when framework is marked not present
        with patch("opensps.torch.autograd._HAS_TORCH", False):
            with self.assertRaises(ImportError):
                opensps.torch.autograd.csp_batch(self.ctx, {"dust2": [0.2, 0.4]})

        with patch("opensps.jax.primitives._HAS_JAX", False):
            with self.assertRaises(ImportError):
                opensps.jax.primitives.csp_batch(self.ctx, {"dust2": [0.2, 0.4]})

    def test_pytorch_autograd_real_or_mock(self):
        try:
            import torch
            has_torch = True
        except ImportError:
            has_torch = False

        if has_torch:
            dust2 = torch.tensor([0.2, 0.5], dtype=torch.float64, requires_grad=True)
            fagn = torch.tensor([0.05, 0.15], dtype=torch.float64, requires_grad=True)

            mags, spec, mass, lbol = opensps.torch.csp_batch(
                self.ctx,
                params={"dust2": dust2, "fagn": fagn}
            )
            self.assertEqual(mags.shape, (2, self.ctx.nbands))
            self.assertEqual(spec.shape, (2, self.ctx.nspec))

            loss = mags.sum()
            loss.backward()

            self.assertIsNotNone(dust2.grad)
            self.assertIsNotNone(fagn.grad)
            self.assertTrue(torch.all(torch.isfinite(dust2.grad)))
            self.assertTrue(torch.all(torch.isfinite(fagn.grad)))

    def test_pytorch_autograd_gpu(self):
        try:
            import torch
            has_torch = True
        except ImportError:
            has_torch = False

        if has_torch and torch.cuda.is_available():
            device = torch.device("cuda:0")
            dust2 = torch.tensor([0.2, 0.5], dtype=torch.float64, device=device, requires_grad=True)
            fagn = torch.tensor([0.05, 0.15], dtype=torch.float64, device=device, requires_grad=True)

            mags, spec, mass, lbol = opensps.torch.csp_batch(
                self.ctx,
                params={"dust2": dust2, "fagn": fagn}
            )
            self.assertEqual(mags.device, device)
            self.assertEqual(spec.device, device)
            self.assertEqual(mags.shape, (2, self.ctx.nbands))
            self.assertEqual(spec.shape, (2, self.ctx.nspec))

            # GPU resident likelihood computation
            target_mags = mags.detach() + 0.01
            loss = torch.sum((mags - target_mags) ** 2)
            loss.backward()

            self.assertIsNotNone(dust2.grad)
            self.assertIsNotNone(fagn.grad)
            self.assertEqual(dust2.grad.device, device)
            self.assertEqual(fagn.grad.device, device)
            self.assertTrue(torch.all(torch.isfinite(dust2.grad)))
            self.assertTrue(torch.all(torch.isfinite(fagn.grad)))

    def test_jax_custom_vjp_real_or_mock(self):
        try:
            import jax
            import jax.numpy as jnp
            has_jax = True
        except ImportError:
            has_jax = False

        if has_jax:
            j_dust2 = jnp.array([0.2, 0.5])
            j_fagn = jnp.array([0.05, 0.15])

            def loss_fn(p_dust, p_fagn):
                m, s, ma, lb = opensps.jax.csp_batch(self.ctx, {"dust2": p_dust, "fagn": p_fagn})
                return jnp.sum(m)

            mags, spec, mass, lbol = opensps.jax.csp_batch(self.ctx, {"dust2": j_dust2, "fagn": j_fagn})
            self.assertEqual(mags.shape, (2, self.ctx.nbands))
            self.assertEqual(spec.shape, (2, self.ctx.nspec))

            grad_fn = jax.grad(loss_fn, argnums=(0, 1))
            g_dust, g_fagn = grad_fn(j_dust2, j_fagn)

            self.assertTrue(jnp.all(jnp.isfinite(g_dust)))
            self.assertTrue(jnp.all(jnp.isfinite(g_fagn)))


if __name__ == "__main__":
    unittest.main()
