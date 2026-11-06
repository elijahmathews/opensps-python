"""Unit tests for opensps.torch and opensps.jax ML integrations."""

import os
import unittest
from unittest.mock import patch
import numpy as np
import opensps
import opensps.torch
import opensps.jax

try:
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False



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

    @unittest.skipUnless(_HAS_JAX, "JAX is not installed")
    def test_jax_custom_vjp_real_or_mock(self):
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

    # ---- JAX transformations (regression: the primitive used to call np.asarray on tracers, so
    # ---- jit, jit+grad and vmap raised TracerArrayConversionError) ----

    def _jax_inputs(self):
        return jnp.array([0.2, 0.5]), jnp.array([0.05, 0.15])

    @unittest.skipUnless(_HAS_JAX, "JAX is not installed")
    def test_jax_jit_matches_eager(self):
        dust2, fagn = self._jax_inputs()

        def f(d, a):
            return opensps.jax.csp_batch(self.ctx, {"dust2": d, "fagn": a})

        eager = f(dust2, fagn)
        jitted = jax.jit(f)(dust2, fagn)
        for e, j in zip(eager, jitted):
            self.assertEqual(e.shape, j.shape)
            np.testing.assert_array_equal(np.asarray(e), np.asarray(j))

    @unittest.skipUnless(_HAS_JAX, "JAX is not installed")
    def test_jax_jit_grad_matches_batch_gradients(self):
        dust2, fagn = self._jax_inputs()

        def loss(d, a):
            return opensps.jax.csp_batch(self.ctx, {"dust2": d, "fagn": a})[0].sum()

        g_dust, g_fagn = jax.jit(jax.grad(loss, argnums=(0, 1)))(dust2, fagn)
        g_eager = jax.grad(loss, argnums=(0, 1))(dust2, fagn)

        ref = self.ctx.batch({"dust2": np.asarray(dust2, dtype=np.float64), "fagn": np.asarray(fagn, dtype=np.float64)})
        ref = ref.compute_gradients(mags_bar=np.ones((2, self.ctx.nbands)))
        for got, eager, name in ((g_dust, g_eager[0], "dust2"), (g_fagn, g_eager[1], "fagn")):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(eager))
            np.testing.assert_allclose(np.asarray(got), ref.params_bar[name], rtol=1e-4)

    @unittest.skipUnless(_HAS_JAX, "JAX is not installed")
    def test_jax_vmap_matches_loop(self):
        tage = jnp.array([3.0, 8.0])

        def f(d):
            return opensps.jax.csp_batch(self.ctx, {"dust2": d, "tage": tage})[0].sum()

        batched = jnp.array([[0.2, 0.5], [0.1, 0.3], [0.6, 0.9]])
        out = jax.vmap(f)(batched)
        self.assertEqual(out.shape, (3,))
        np.testing.assert_allclose(np.asarray(out), np.array([float(f(b)) for b in batched]), rtol=1e-6)

        # and through the reverse pass
        g = jax.vmap(jax.grad(f))(batched)
        self.assertEqual(g.shape, batched.shape)
        np.testing.assert_allclose(np.asarray(g[1]), np.asarray(jax.grad(f)(batched[1])), rtol=1e-6)

    @unittest.skipUnless(_HAS_JAX, "JAX is not installed")
    def test_jax_scalar_parameter_gradient_is_sum_over_models(self):
        tage = jnp.array([3.0, 5.0, 8.0])

        g_scalar = jax.grad(lambda d: opensps.jax.csp_batch(self.ctx, {"dust2": d, "tage": tage})[0].sum())(jnp.float32(0.3))
        g_vector = jax.grad(
            lambda d: opensps.jax.csp_batch(self.ctx, {"dust2": d, "tage": tage})[0].sum()
        )(jnp.full((3,), 0.3, dtype=jnp.float32))
        self.assertEqual(g_scalar.shape, ())
        np.testing.assert_allclose(float(g_scalar), float(g_vector.sum()), rtol=1e-5)

    @unittest.skipUnless(_HAS_JAX, "JAX is not installed")
    def test_jax_ssp_weights_under_jit(self):
        rng = np.random.default_rng(0)
        w = np.abs(rng.normal(size=(2, self.ctx.ntfull)))
        w = jnp.asarray(w / w.sum(axis=1, keepdims=True), dtype=jnp.float32)
        d = jnp.array([0.2, 0.4])
        gw = jax.jit(jax.grad(lambda w_: opensps.jax.csp_batch(self.ctx, {"dust2": d}, ssp_weights=w_)[0].sum()))(w)
        self.assertEqual(gw.shape, w.shape)
        self.assertTrue(bool(jnp.all(jnp.isfinite(gw))))


if __name__ == "__main__":
    unittest.main()
