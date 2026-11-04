# opensps

Modern, high-throughput Python interface for [libopensps](https://github.com/elijahmathews/libopensps), a modernized and accelerated Flexible Stellar Population Synthesis (FSPS) library.

## Key Features

- **High-Throughput Batched Calculations (`opensps.CSPBatch`)**: Vectorized forward model over thousands of models on GPU and CPU.
- **Reverse-Mode Analytical Gradients (`compute_gradients`)**: Exact adjoint derivatives with respect to 33 physical parameters, non-parametric SFH weights, and 3D enrichment histories in a fraction of the time of finite differences.
- **Batched SSP Operators (`opensps.SSPOperator`)**: Decouples isochrones from IMF integrals, allowing IMF slopes, characteristic masses, and bounds to vary freely per galaxy in batches, or fast direct evaluation.
- **Drop-in `python-fsps` Compatibility (`opensps.StellarPopulation`)**: Near drop-in compatibility with legacy `fsps.StellarPopulation` for seamless integration into Prospector and existing astronomy pipelines.
- **PyTorch Autograd Integration (`opensps.torch.csp_batch`)**: Native differentiable forward model (`torch.autograd.Function`) for deep learning, simulation-based inference (SBI), and Hamiltonian Monte Carlo (HMC/NUTS).

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

### 5. Differentiable PyTorch Modeling

```python
import torch
import opensps
import opensps.torch as sytorch

ctx = opensps.Context(zmet=-1, isoc="mist", spec="miles", dust="DL07")

dust2 = torch.tensor([0.2, 0.5], requires_grad=True, dtype=torch.float64)
tage = torch.tensor([5.0, 10.0], requires_grad=True, dtype=torch.float64)

mags, spec, mass, lbol = sytorch.csp_batch(ctx, {"dust2": dust2, "tage": tage})

loss = mags.sum()
loss.backward()

print("PyTorch dust2 gradient:", dust2.grad)
```

## Running Tests

```bash
python3 -m unittest discover -s tests
```
