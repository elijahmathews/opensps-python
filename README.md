<p align="center">
  <img src="assets/logo.svg" alt="OpenSPS logo" width="180px" height="180px" />
</p>

<h1 align="center">opensps</h1>

<p align="center">
  <strong>High-Throughput, Hardware-Accelerated Stellar Population Synthesis for Python</strong>
</p>

<p align="center">
  <a href="LICENSE.md"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/github/actions/workflow/status/elijahmathews/opensps-python/ci.yml" alt="GitHub CI Status">
  <img src="https://img.shields.io/codecov/c/github/elijahmathews/opensps-python" alt="Codecov Status">
</p>

Modern, high-throughput Python interface for [libopensps](https://github.com/elijahmathews/libopensps), a modernized and accelerated Flexible Stellar Population Synthesis (FSPS) library.

## Key Features

- **High-Throughput Batched Calculations (`opensps.CSPBatch`)**: Vectorized forward model over thousands of models on GPU and CPU.
- **Reverse-Mode Analytical Gradients (`compute_gradients`)**: Exact adjoint derivatives with respect to 33 physical parameters, non-parametric SFH weights, and 3D enrichment histories in a fraction of the time of finite differences.
- **Batched SSP Operators (`opensps.SSPOperator`)**: Decouples isochrones from IMF integrals, allowing IMF slopes, characteristic masses, and bounds to vary freely per galaxy in batches, or fast direct evaluation.
- **Drop-in `python-fsps` Compatibility (`opensps.StellarPopulation`)**: Near drop-in compatibility with legacy `fsps.StellarPopulation` for seamless integration into Prospector and existing astronomy pipelines.
- **PyTorch Autograd Integration (`opensps.torch.csp_batch`)**: Native differentiable forward model (`torch.autograd.Function`) for deep learning, simulation-based inference (SBI), and Hamiltonian Monte Carlo (HMC/NUTS).
- **JAX `custom_vjp` Integration (`opensps.jax.csp_batch`)**: Differentiable vector-Jacobian products compatible with JAX transformations (`jax.grad`, `jax.vjp`).
- **Zero-Copy GPU Memory Access**: Retain computed spectral matrices on accelerator VRAM (e.g. AMD Radeon/ROCm or NVIDIA GPUs) to minimize PCIe crosstalk in GPU-resident likelihoods.

## Quickstart

### 1. High-Throughput Batch Forward Modeling

```python
import opensps
import numpy as np

# Create an isolated calculation context
ctx = opensps.Context(zmet=-1, isoc="mist", spec="miles", dust="DL07")

# Configure batch parameters for N = 1,000 models
N = 1000
params = {
    "dust2": np.random.uniform(0.1, 1.0, size=N),
    "tage": np.random.uniform(1.0, 12.0, size=N),
    "tau": 2.0,  # Scalars automatically broadcast across models
}

batch = ctx.batch(params)
res = batch.compute(compute_spec=True)

print("Photometry shape:", res.mags.shape)   # (1000, nbands)
print("Spectra shape:", res.spec.shape)       # (1000, nspec)
print("Surviving stellar mass:", res.mass)     # (1000,)
```

### Non-parametric SFH weights

```python
from opensps import sfh

# seven lookback-time bins per galaxy -> a (N, ntfull) SSP-weight matrix for ``ctx.batch``
t, s = sfh.step_table(edges_yr, sfr_per_bin)           # edges (N, 8) [yr], sfr (N, 7) [M_sun/yr]
weights = sfh.ssp_weights_from_table(ctx.log_ages, t, s)
res = ctx.batch({"dust2": dust2}, ssp_weights=weights / weights.sum(1, keepdims=True)).compute()
```

The weights are those of libopensps's tabular SFH (hats linear in log age), vectorised over models.

### 2. Analytical Adjoint Gradients

```python
# Provide cotangents of your loss (e.g., dL / d(mags))
mags_bar = np.ones((N, ctx.nbands), dtype=np.float64)

grad_res = batch.compute_gradients(mags_bar=mags_bar)

# Exact analytical derivatives with zero finite-difference noise:
print("dL / d(dust2):", grad_res.params_bar["dust2"])
print("dL / d(tage):", grad_res.params_bar["tage"])
```

### 3. Free IMF Parameters per Model via SSP Operators

```python
# Enable SSP operator on context for Kroupa IMF
ctx["imf_type"] = 2
ctx.build_ssp_operator(all_metallicities=False)

# Kroupa slopes (imf1, imf2, imf3) can now vary freely per galaxy!
params = {
    "dust2": [0.2, 0.5, 0.8],
    "imf1": [1.1, 1.3, 1.5],
    "imf2": [2.1, 2.3, 2.5],
    "tage": 10.0,
}

res = ctx.batch(params).compute()
print("Batch computed with free IMFs:", res.mags.shape)
```

### 4. Drop-in `python-fsps` Compatibility

```python
from opensps import StellarPopulation

sp = StellarPopulation(compute_vega_mags=False, zcontinuous=1)
sp.params["logzsol"] = -0.3
sp.params["dust2"] = 0.5

wave, spec = sp.get_spectrum(tage=10.0)
mags = sp.get_mags(tage=10.0)
```

### 5. Differentiable PyTorch & JAX Modeling

#### PyTorch Autograd:
```python
import torch
import opensps
import opensps.torch as sytorch

ctx = opensps.Context(zmet=-1, isoc="mist", spec="miles", dust="DL07")
ctx["tage"] = 10.0

dust2 = torch.tensor([0.2, 0.5], requires_grad=True, dtype=torch.float64)
fagn = torch.tensor([0.05, 0.15], requires_grad=True, dtype=torch.float64)

mags, spec, mass, lbol = sytorch.csp_batch(ctx, {"dust2": dust2, "fagn": fagn})

loss = mags.sum()
loss.backward()

print("dL / d(dust2):", dust2.grad)
```

#### JAX `custom_vjp`:
```python
import jax
import jax.numpy as jnp
import opensps
import opensps.jax as syjax

ctx = opensps.Context(zmet=-1, isoc="mist", spec="miles", dust="DL07")
ctx["tage"] = 10.0

def loss_fn(p_dust, p_fagn):
    mags, spec, mass, lbol = syjax.csp_batch(ctx, {"dust2": p_dust, "fagn": p_fagn})
    return jnp.sum(mags)

j_dust2 = jnp.array([0.2, 0.5])
j_fagn = jnp.array([0.05, 0.15])

grad_fn = jax.grad(loss_fn, argnums=(0, 1))
g_dust, g_fagn = grad_fn(j_dust2, j_fagn)
print("JAX dL / d(dust2):", g_dust)
```

### 6. Zero-Copy GPU Memory Access (Minimizing Host Crosstalk)

When `libopensps` is compiled with GPU target offload (e.g. on AMD ROCm or NVIDIA CUDA), the computed batch spectra remain directly in GPU VRAM:

```python
# Pass compute_spec=False or spec=None to compute without PCIe host transfers
batch = ctx.batch(params)
res = batch.compute(compute_spec=False)

# Retrieve raw GPU memory address of the spectral matrix [nspec x n_models]
gpu_addr, nspec, n_models, bytes_per_elem = ctx.get_batch_spec_device_ptr()
print(f"Device VRAM address: 0x{gpu_addr:x} ({nspec} x {n_models})")
```

## Running Tests

```bash
python3 -m unittest discover -s tests
```
