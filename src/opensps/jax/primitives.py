"""JAX custom_vjp integration for OpenSPS.

The forward and reverse passes call libopensps on the host through ``jax.pure_callback``, so
``csp_batch`` works under ``jax.jit``, ``jax.grad``, ``jax.value_and_grad`` and ``jax.vmap`` (a vmapped
call runs the host batch once per mapped element, ``vmap_method="sequential"``). Inputs may live on any
device; they are copied to the host for the library call and the results copied back.
"""

try:
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

from typing import Dict, Optional, Tuple, Any
import numpy as np


if _HAS_JAX:
    def _batch_size(context: Any, weights_shape: Optional[Tuple[int, ...]], param_shapes) -> int:
        """Number of models N, from the (static) shapes of the parameters and the SSP weights."""
        n = 1
        for shp in param_shapes:
            if len(shp) > 0:
                n = max(n, int(shp[0]))
        if weights_shape is not None and len(weights_shape) == 2:
            nt = context.ntfull
            if weights_shape[1] == nt:
                n = max(n, int(weights_shape[0]))
            elif weights_shape[0] == nt:
                n = max(n, int(weights_shape[1]))
        return n

    def _create_vjp_primitive(context: Any, param_names: Tuple[str, ...], has_weights: bool):
        def _host_batch(weights, params):
            np_params = {name: np.asarray(arr, dtype=np.float64) for name, arr in zip(param_names, params)}
            np_weights = np.asarray(weights, dtype=np.float64) if has_weights else None
            return context.batch(params=np_params, ssp_weights=np_weights)

        def _forward_host(weights, *params):
            res = _host_batch(weights, params).compute(compute_spec=True)
            dt = jax.dtypes.canonicalize_dtype(np.float64)
            return (
                np.asarray(res.mags, dtype=dt),
                np.asarray(res.spec, dtype=dt),
                np.asarray(res.mass, dtype=dt),
                np.asarray(res.lbol, dtype=dt),
            )

        def _backward_host(weights, *rest):
            params = rest[:len(param_names)]
            grad_mags, grad_spec, grad_mass, grad_lbol = rest[len(param_names):]
            batch = _host_batch(weights, params)
            grad_res = batch.compute_gradients(
                mags_bar=np.asarray(grad_mags, dtype=np.float64),
                spec_bar=np.asarray(grad_spec, dtype=np.float64),
                mass_bar=np.asarray(grad_mass, dtype=np.float64),
                lbol_bar=np.asarray(grad_lbol, dtype=np.float64),
            )
            out = []
            w_arr = np.asarray(weights)
            if has_weights and grad_res.weights_bar is not None:
                wb = np.asarray(grad_res.weights_bar, dtype=np.float64)
                out.append(wb.reshape(w_arr.shape) if wb.size == w_arr.size else wb.T.reshape(w_arr.shape))
            else:
                out.append(np.zeros(w_arr.shape, dtype=w_arr.dtype))
            for name, arr in zip(param_names, params):
                arr = np.asarray(arr)
                g = grad_res.params_bar.get(name)
                if g is None:
                    out.append(np.zeros(arr.shape, dtype=arr.dtype))
                elif arr.size == 1:
                    # A scalar broadcast over the batch: its gradient is the sum over the models
                    out.append(np.asarray(np.sum(g), dtype=arr.dtype).reshape(arr.shape))
                else:
                    out.append(np.asarray(g, dtype=arr.dtype).reshape(arr.shape))
            return tuple(out)

        def _structs(weights, params):
            n = _batch_size(context, tuple(weights.shape) if has_weights else None, [p.shape for p in params])
            dt = jax.dtypes.canonicalize_dtype(np.float64)
            return (
                jax.ShapeDtypeStruct((n, context.nbands), dt),
                jax.ShapeDtypeStruct((n, context.nspec), dt),
                jax.ShapeDtypeStruct((n,), dt),
                jax.ShapeDtypeStruct((n,), dt),
            )

        @jax.custom_vjp
        def _prim(weights_or_none, *param_arrays):
            return jax.pure_callback(
                _forward_host, _structs(weights_or_none, param_arrays),
                weights_or_none, *param_arrays, vmap_method="sequential",
            )

        def _fwd(weights_or_none, *param_arrays):
            out = _prim(weights_or_none, *param_arrays)
            # Only JAX arrays or PyTrees of arrays may be stored in residual
            return out, (weights_or_none, param_arrays)

        def _bwd(residual, cotangents):
            weights_or_none, param_arrays = residual
            shapes = (jax.ShapeDtypeStruct(weights_or_none.shape, weights_or_none.dtype),) + tuple(
                jax.ShapeDtypeStruct(p.shape, p.dtype) for p in param_arrays
            )
            grads = jax.pure_callback(
                _backward_host, shapes,
                weights_or_none, *param_arrays, *cotangents, vmap_method="sequential",
            )
            return tuple(grads)

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

        Works under ``jax.jit``, ``jax.grad`` and ``jax.vmap``. The output dtype follows JAX's
        default float precision (float32 unless ``jax_enable_x64`` is set).

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
        param_arrays = tuple(jnp.asarray(params[k]) for k in param_names)
        has_weights = ssp_weights is not None
        dummy_weights = jnp.asarray(ssp_weights) if has_weights else jnp.zeros(1)
        prim = _get_primitive(context, param_names, has_weights)
        return prim(dummy_weights, *param_arrays)

else:
    def csp_batch(*args, **kwargs):
        raise ImportError("JAX is not installed. Please install jax to use opensps.jax.")
