"""JAX custom_vjp integration for OpenSPS."""

try:
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

from typing import Dict, Optional, Tuple, Any
import numpy as np


if _HAS_JAX:
    def _create_vjp_primitive(context: Any, param_names: Tuple[str, ...], has_weights: bool):
        @jax.custom_vjp
        def _prim(weights_or_none, *param_arrays):
            np_params = {name: np.asarray(arr, dtype=np.float64) for name, arr in zip(param_names, param_arrays)}
            np_weights = np.asarray(weights_or_none, dtype=np.float64) if has_weights else None
            batch = context.batch(params=np_params, ssp_weights=np_weights)
            res = batch.compute(compute_spec=True)
            return (
                jnp.asarray(res.mags),
                jnp.asarray(res.spec),
                jnp.asarray(res.mass),
                jnp.asarray(res.lbol),
            )

        def _fwd(weights_or_none, *param_arrays):
            out = _prim(weights_or_none, *param_arrays)
            # Only JAX arrays or PyTrees of arrays may be stored in residual
            residual = (weights_or_none if has_weights else None, param_arrays)
            return out, residual

        def _bwd(residual, cotangents):
            weights_or_none, param_arrays = residual
            grad_mags, grad_spec, grad_mass, grad_lbol = cotangents

            np_params = {name: np.asarray(arr, dtype=np.float64) for name, arr in zip(param_names, param_arrays)}
            np_weights = np.asarray(weights_or_none, dtype=np.float64) if has_weights else None
            batch = context.batch(params=np_params, ssp_weights=np_weights)

            m_bar = np.asarray(grad_mags, dtype=np.float64) if grad_mags is not None else None
            s_bar = np.asarray(grad_spec, dtype=np.float64) if grad_spec is not None else None
            mass_bar = np.asarray(grad_mass, dtype=np.float64) if grad_mass is not None else None
            lbol_bar = np.asarray(grad_lbol, dtype=np.float64) if grad_lbol is not None else None

            grad_res = batch.compute_gradients(
                mags_bar=m_bar,
                spec_bar=s_bar,
                mass_bar=mass_bar,
                lbol_bar=lbol_bar,
            )

            weights_grad = (
                jnp.asarray(grad_res.weights_bar)
                if has_weights and grad_res.weights_bar is not None
                else (jnp.zeros_like(weights_or_none) if has_weights else None)
            )

            param_grads = []
            for name, arr in zip(param_names, param_arrays):
                g_arr = grad_res.params_bar.get(name)
                if g_arr is not None:
                    param_grads.append(jnp.asarray(g_arr, dtype=arr.dtype))
                else:
                    param_grads.append(jnp.zeros_like(arr))

            return (weights_grad, *param_grads)

        _prim.defvjp(_fwd, _bwd)
        return _prim

    _PRIM_CACHE = {}

    def _get_primitive(context: Any, param_names: Tuple[str, ...], has_weights: bool):
        key = (context.handle, param_names, has_weights)
        if key not in _PRIM_CACHE:
            _PRIM_CACHE[key] = _create_vjp_primitive(context, param_names, has_weights)
        return _PRIM_CACHE[key]

    def csp_batch(
        context: Any,
        params: Dict[str, Any],
        ssp_weights: Optional[Any] = None,
    ) -> Tuple[Any, Any, Any, Any]:
        """Differentiable JAX operator for batched CSP calculation using custom_vjp.

        Parameters
        ----------
        context : opensps.Context
            Calculation context.
        params : dict of str -> jax.Array
            Differentiable parameter arrays.
        ssp_weights : jax.Array, optional
            Non-parametric SFH weight matrix (N, ntfull).

        Returns
        -------
        mags : jax.Array of shape (N, nbands)
        spec : jax.Array of shape (N, nspec)
        mass : jax.Array of shape (N,)
        lbol : jax.Array of shape (N,)
        """
        if not _HAS_JAX:
            raise ImportError("JAX is not installed. Please install jax to use opensps.jax.")

        param_names = tuple(params.keys())
        param_arrays = tuple(params[k] for k in param_names)
        has_weights = ssp_weights is not None
        dummy_weights = ssp_weights if has_weights else jnp.zeros(1)
        prim = _get_primitive(context, param_names, has_weights)
        return prim(dummy_weights, *param_arrays)

else:
    def csp_batch(*args, **kwargs):
        raise ImportError("JAX is not installed. Please install jax to use opensps.jax.")
