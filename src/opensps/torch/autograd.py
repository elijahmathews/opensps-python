"""PyTorch autograd integration for OpenSPS."""

try:
    import torch
    from torch.autograd import Function
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from typing import Dict, Optional, Tuple, Any
import numpy as np


if _HAS_TORCH:
    class _CSPBatchAutograd(Function):
        @staticmethod
        def forward(
            ctx_obj,
            context,
            param_names: Tuple[str, ...],
            ssp_weights_tensor: Optional[torch.Tensor],
            *param_tensors: torch.Tensor,
        ):
            # Save metadata for backward pass
            ctx_obj.context = context
            ctx_obj.param_names = param_names
            ctx_obj.has_weights = ssp_weights_tensor is not None

            # Convert PyTorch tensors to NumPy
            np_params = {
                name: tensor.detach().cpu().numpy()
                for name, tensor in zip(param_names, param_tensors)
            }
            np_weights = (
                ssp_weights_tensor.detach().cpu().numpy()
                if ssp_weights_tensor is not None
                else None
            )

            batch = context.batch(params=np_params, ssp_weights=np_weights)
            res = batch.compute(compute_spec=True)

            ctx_obj.batch = batch
            mags_t = torch.from_numpy(res.mags).to(param_tensors[0].device)
            spec_t = torch.from_numpy(res.spec).to(param_tensors[0].device)
            mass_t = torch.from_numpy(res.mass).to(param_tensors[0].device)
            lbol_t = torch.from_numpy(res.lbol).to(param_tensors[0].device)

            return mags_t, spec_t, mass_t, lbol_t

        @staticmethod
        def backward(ctx_obj, grad_mags, grad_spec, grad_mass, grad_lbol):
            context = ctx_obj.context
            batch = ctx_obj.batch
            param_names = ctx_obj.param_names

            # Convert cotangents to NumPy
            m_bar = grad_mags.detach().cpu().numpy() if grad_mags is not None else None
            s_bar = grad_spec.detach().cpu().numpy() if grad_spec is not None else None
            mass_bar = grad_mass.detach().cpu().numpy() if grad_mass is not None else None
            lbol_bar = grad_lbol.detach().cpu().numpy() if grad_lbol is not None else None

            grad_res = batch.compute_gradients(
                mags_bar=m_bar,
                spec_bar=s_bar,
                mass_bar=mass_bar,
                lbol_bar=lbol_bar,
            )

            # Map parameter gradients back to PyTorch tensors in the order received
            param_grads = []
            for name in param_names:
                g_arr = grad_res.params_bar.get(name)
                if g_arr is not None:
                    g_tensor = torch.from_numpy(g_arr)
                    param_grads.append(g_tensor)
                else:
                    param_grads.append(None)

            weights_grad = None
            if ctx_obj.has_weights and grad_res.weights_bar is not None:
                weights_grad = torch.from_numpy(grad_res.weights_bar)

            # Returns grads for: context, param_names, ssp_weights_tensor, *param_tensors
            return (None, None, weights_grad, *param_grads)


    def csp_batch(
        context: "Context",
        params: Dict[str, torch.Tensor],
        ssp_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Differentiable PyTorch operator for batched CSP calculation.

        Parameters
        ----------
        context : opensps.Context
            Calculation context.
        params : dict of str -> torch.Tensor
            Differentiable parameter tensors (e.g. 'dust2', 'logzsol', 'tage').
        ssp_weights : torch.Tensor, optional
            Non-parametric SFH weight matrix (N, ntfull).

        Returns
        -------
        mags : torch.Tensor of shape (N, nbands)
        spec : torch.Tensor of shape (N, nspec)
        mass : torch.Tensor of shape (N,)
        lbol : torch.Tensor of shape (N,)
        """
        param_names = tuple(params.keys())
        param_tensors = tuple(params.values())

        return _CSPBatchAutograd.apply(context, param_names, ssp_weights, *param_tensors)

else:
    def csp_batch(*args, **kwargs):
        raise ImportError("PyTorch is not installed. Please install torch to use opensps.torch.")
