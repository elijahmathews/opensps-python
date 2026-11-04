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
    def _csp_batch_fwd(param_names, context, has_weights, ssp_weights, *param_arrays):
        np_params = {name: np.asarray(arr) for name, arr in zip(param_names, param_arrays)}
        np_weights = np.asarray(ssp_weights) if has_weights else None
        batch = context.batch(params=np_params, ssp_weights=np_weights)
        res = batch.compute(compute_spec=True)
        out = (
            jnp.asarray(res.mags),
            jnp.asarray(res.spec),
            jnp.asarray(res.mass),
            jnp.asarray(res.lbol),
        )
        residual = (param_names, context, has_weights, batch)
        return out, residual

    def _csp_batch_bwd(residual, cotangents):
        param_names, context, has_weights, batch = residual
        grad_mags, grad_spec, grad_mass, grad_lbol = cotangents

        m_bar = np.asarray(grad_mags) if grad_mags is not None else None
        s_bar = np.asarray(grad_spec) if grad_spec is not None else None
        mass_bar = np.asarray(grad_mass) if grad_mass is not None else None
        lbol_bar = np.asarray(grad_lbol) if grad_lbol is not None else None

        grad_res = batch.compute_gradients(
            mags_bar=m_bar,
            spec_bar=s_bar,
            mass_bar=mass_bar,
            lbol_bar=lbol_bar,
        )

        weights_grad = (
            jnp.asarray(grad_res.weights_bar)
            if has_weights and grad_res.weights_bar is not None
            else None
        )

        param_grads = []
        for name in param_names:
            g_arr = grad_res.params_bar.get(name)
            if g_arr is not None:
                param_grads.append(jnp.asarray(g_arr))
            else:
                param_grads.append(None)

        return (None, None, None, weights_grad, *param_grads)

    @jax.custom_vjp
    def _csp_batch_prim(param_names, context, has_weights, ssp_weights, *param_arrays):
        np_params = {name: np.asarray(arr) for name, arr in zip(param_names, param_arrays)}
        np_weights = np.asarray(ssp_weights) if has_weights else None
        batch = context.batch(params=np_params, ssp_weights=np_weights)
        res = batch.compute(compute_spec=True)
        return (
            jnp.asarray(res.mags),
            jnp.asarray(res.spec),
            jnp.asarray(res.mass),
            jnp.asarray(res.lbol),
        )

    _csp_batch_prim.defvjp(_csp_batch_fwd, _csp_batch_bwd)

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
        param_names = tuple(params.keys())
        param_arrays = tuple(params[k] for k in param_names)
        has_weights = ssp_weights is not None
        return _csp_batch_prim(param_names, context, has_weights, ssp_weights, *param_arrays)

else:
    def csp_batch(*args, **kwargs):
        raise ImportError("JAX is not installed. Please install jax to use opensps.jax.")
