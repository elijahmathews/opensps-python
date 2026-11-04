"""Unit tests for opensps.torch and opensps.jax ML integrations."""

import os
import sys
import unittest
import importlib
from unittest.mock import MagicMock, patch
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

    @classmethod
    def tearDownClass(cls):
        cls.ctx.destroy()

    def test_missing_dependencies_raise_error(self):
        # When neither torch nor jax is installed, calling csp_batch raises ImportError
        with self.assertRaises(ImportError):
            opensps.torch.csp_batch(self.ctx, {"dust2": [0.2, 0.4]})

        with self.assertRaises(ImportError):
            opensps.jax.csp_batch(self.ctx, {"dust2": [0.2, 0.4]})

    def test_torch_autograd_mocked_execution(self):
        mock_torch = MagicMock()
        mock_autograd = MagicMock()
        mock_autograd.Function = object
        mock_torch.autograd = mock_autograd
        mock_torch.from_numpy = lambda arr: DummyTensor(arr)

        class DummyTensor:
            def __init__(self, arr, device="cpu"):
                self.arr = np.asarray(arr)
                self.device = device
            def detach(self):
                return self
            def cpu(self):
                return self
            def numpy(self):
                return self.arr
            def to(self, dev):
                return self

        with patch.dict("sys.modules", {"torch": mock_torch, "torch.autograd": mock_autograd}):
            import opensps.torch.autograd as syautograd
            importlib.reload(syautograd)

            ctx_obj = MagicMock()
            param_names = ("dust2",)
            t_dust = DummyTensor([0.2, 0.5])

            # Test forward execution
            mags_t, spec_t, mass_t, lbol_t = syautograd._CSPBatchAutograd.forward(
                ctx_obj, self.ctx, param_names, None, t_dust
            )
            self.assertIsNotNone(mags_t)
            self.assertIsNotNone(spec_t)

            # Test backward execution
            grad_mags = DummyTensor(np.ones((2, self.ctx.nbands)))
            ret = syautograd._CSPBatchAutograd.backward(
                ctx_obj, grad_mags, None, None, None
            )
            self.assertEqual(len(ret), 4)

            # Test csp_batch wrapper
            syautograd._CSPBatchAutograd.apply = MagicMock(return_value=(mags_t, spec_t, mass_t, lbol_t))
            res = syautograd.csp_batch(self.ctx, {"dust2": t_dust})
            self.assertEqual(len(res), 4)

        # Restore
        importlib.reload(syautograd)

    def test_jax_primitives_mocked_execution(self):
        def mock_custom_vjp(fn):
            fn.defvjp = lambda fwd, bwd: None
            return fn

        mock_jax = MagicMock()
        mock_jax.custom_vjp = mock_custom_vjp
        mock_jnp = MagicMock()
        mock_jnp.asarray = lambda arr: np.asarray(arr)

        with patch.dict("sys.modules", {"jax": mock_jax, "jax.numpy": mock_jnp}):
            import opensps.jax.primitives as syjax
            importlib.reload(syjax)

            param_names = ("dust2",)
            t_dust = np.array([0.2, 0.5])

            # Test forward execution
            out, residual = syjax._csp_batch_fwd(param_names, self.ctx, False, None, t_dust)
            self.assertEqual(len(out), 4)

            # Test backward execution
            cotangents = (np.ones((2, self.ctx.nbands)), None, None, None)
            grads = syjax._csp_batch_bwd(residual, cotangents)
            self.assertEqual(len(grads), 5)

            # Test csp_batch wrapper
            syjax._csp_batch_prim = MagicMock(return_value=out)
            res = syjax.csp_batch(self.ctx, {"dust2": t_dust})
            self.assertEqual(len(res), 4)

        # Restore
        importlib.reload(syjax)


if __name__ == "__main__":
    unittest.main()
