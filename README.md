# opensps

Python bindings for libopensps, a modernized FSPS (Flexible Stellar Population Synthesis).

- `opensps.Context`: the library's context API. Parameters by name, the SSP grid, single
  composite populations, and the batched CSP that evaluates many models in one call, with one
  NumPy array per parameter. Arrays go in and come out as NumPy arrays that the library writes
  directly; computations release the GIL.
- `opensps.StellarPopulation`: a layer compatible with python-fsps, so existing scripts and
  Prospector run with few changes (see [docs/compatibility.md](docs/compatibility.md)).

Status: in development. The C API entry points it needs are in the core's
`capi/bindings-support` branch (elijahmathews/fsps#20).

## Install

The package needs a C and a Fortran compiler (gfortran); Meson, Ninja, Cython and NumPy are
pulled in by pip. The core library comes from one of two sources:

**(a) Built with the package (default).** If no installed core is found, `pip install .`
fetches the core repository as a Meson subproject (`subprojects/fsps.wrap`, pinned to a
commit), builds it as a static library without its tests or data, and links it into the
extension:

```sh
pip install .
# force this path even if a core library is installed:
pip install . -Csetup-args=--wrap-mode=forcefallback
```

Options of the core pass through, e.g. FP32 kernels:
`pip install . -Csetup-args=-Dfsps:kernel_precision=single`. To build against a local checkout
of the core instead of the pinned commit, place it (or a symbolic link to it) at
`subprojects/fsps`. (While the core repository still carries its data, the subproject fetch
downloads about 1.3 GB.)

**(b) An installed core library**, found through pkg-config (`opensps.pc`, then `fsps.pc`;
the names are the `core_names` option in `meson.options`):

```sh
# in the core repository
meson setup build --buildtype=release --prefix=$HOME/.local -Dinstall_data=false
meson install -C build
# here
PKG_CONFIG_PATH=$HOME/.local/lib64/pkgconfig pip install . -Csetup-args=--wrap-mode=nofallback
```

The extension records the library directory as its run-time search path. GPU builds of the
core (OpenACC or OpenMP offload) are used the same way: the bindings only call the C API.

Development: in a virtual environment with `meson-python`, `meson`, `ninja`, `cython`, `numpy`
and `pytest`, run `pip install --no-build-isolation .` (or `-e .`), then `pytest`.

## Data

The library reads its data (isochrones, spectral libraries, dust, nebular and filter files)
from a data root, the directory that contains `data/`. Point the package at it with

```python
import opensps
opensps.set_data_home("/path/to/fsps")   # or OPENSPS_DATA_HOME=/path/to/fsps
```

Otherwise the library's own search applies (`SPS_HOME`, `FSPS_DATA_HOME`,
`$XDG_DATA_HOME/fsps`, `~/.local/share/fsps`, `/usr/share/fsps`). `SPS_HOME` wins over
`FSPS_DATA_HOME` there; `set_data_home` sets both.

## Quick start

```python
import opensps

opensps.set_data_home("/path/to/fsps")
ctx = opensps.Context("mist", "miles", "DL07")        # libraries; loads all metallicities
ctx.set_params(sfh=4, tau=2.0, tage=5.0, logzsol=-0.3, dust_type=2, dust2=0.4,
               add_neb_emission=1)

out = ctx.compute_csp(zcontinuous=1)                  # one model, interpolated in log Z
wave = ctx.wavelengths                                # A
spec = out["spec"][0]                                 # L_sun/Hz per M_sun formed
r = out["mags"][0][opensps.get_filter("sdss_r").index]
print(out["mass"][0], out["lbol"][0], r)
```

`compute_csp` returns `age, mass, lbol, sfr, mdust, mformed` (length `n_out`), `spec`
(`n_out x nspec`), `mags` (`n_out x nbands`), `emlines` (`n_out x nemline`) and `indx`;
`n_out` is 1 for `tage > 0` and the number of SSP ages for `tage = 0`. SSP grids are kept
between calls until an SSP parameter changes. `compute_ssp()` returns the SSP grid (`spec` of
shape `ntfull x nspec`, `mass`, `lbol`) at `zmet`.

## Batch

Many models in one call: the context's parameters, with some of them varied per model.

```python
import numpy as np

n = 10_000
rng = np.random.default_rng(1)
ctx.set_params(sfh=4, add_neb_emission=1, add_igm_absorption=1, zred=0.5)
params = {
    "logzsol": rng.uniform(-1.5, 0.2, n),
    "dust2": rng.uniform(0.0, 1.5, n),
    "tau": rng.uniform(0.3, 10.0, n),
    "tage": rng.uniform(1.0, 8.0, n),
    "gas_logu": rng.uniform(-3.5, -1.5, n),
}
res = ctx.compute_csp_batch(params)      # mags (n, nbands), spec (n, nspec), mass, lbol, ...

# In a sampler: write into the same arrays again (no allocation per call)
res = ctx.compute_csp_batch(params, out=res)

# Non-parametric SFHs: SSP weights (n, ntfull) instead of the SFH parameters
w = rng.dirichlet(np.ones(ctx.ntfull), size=n)
res = ctx.compute_csp_batch({"dust2": params["dust2"], "tage": 13.0}, ssp_weights=w,
                            spec=False, emlines=True)
```

Only parameters that act after the SSPs may vary per model (SFH, dust, dust emission,
nebular gas with `add_neb_emission = 1`, AGN, `logzsol`, `zred`, `igm_factor`, ...);
varying one that changes the SSPs (IMF, isochrone options) raises `FSPSError` (status 2):
change it on the context between batches instead. `ctx.batch_spectra_host()` returns the last
batch's spectra in kernel precision (float32 in FP32-kernel builds) without conversion;
`ctx.batch_spectra_device()` gives their device address with `__cuda_array_interface__` in GPU
builds (for CuPy, Numba or PyTorch).

## python-fsps compatibility

```python
sp = opensps.StellarPopulation(zcontinuous=1, sfh=4, tau=2.0, logzsol=-0.3)
wave, spec = sp.get_spectrum(tage=5.0, peraa=True)
mags = sp.get_mags(tage=5.0, bands=["sdss_g", "sdss_r"])
print(sp.stellar_mass, sp.formed_mass, sp.log_lbol, sp.emline_luminosity.shape)
```

For Prospector, call `opensps.install_as_fsps()` before importing `prospect`; `import fsps`
then returns this package. Differences from python-fsps: [docs/compatibility.md](docs/compatibility.md).

## Errors and threads

C status codes become `opensps.FSPSError` (with `.status`); unknown parameter names raise
`opensps.UnknownParameterError` (also a `KeyError`). Missing data raises
`opensps.DataNotFoundError` before the library is called. Contexts are independent: use one
per thread (a context is locked while it computes, and computations release the GIL).

## Tests

```sh
OPENSPS_DATA_HOME=/path/to/fsps pytest
```

The suite compares the bindings bit for bit with direct calls of the same C library through
ctypes, the batch with batches of one and with single CSPs, and `StellarPopulation` with
`Context`. Two optional tests run when their packages are importable: Prospector's
`CSPSpecBasis` / `FastStepBasis` on this package, and a comparison with python-fsps.
`benchmarks/bench_batch.py` times the batch from Python and through ctypes.

## License

MIT, see [LICENSE.md](LICENSE.md).
