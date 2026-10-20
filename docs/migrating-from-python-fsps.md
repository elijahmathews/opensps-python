# Migrating from python-fsps

opensps has its own API (`opensps.Context`, its batch call, NumPy arrays); it is not bound to
python-fsps. To help existing code move over, `opensps.compat.fsps` provides a
`StellarPopulation` that follows python-fsps. This layer is optional and frozen at its current
scope: it gets no new features for python-fsps parity. Port code to `Context` when you can;
the table at the end maps the common calls.

`opensps.compat.fsps.StellarPopulation` follows python-fsps (`fsps.StellarPopulation`): the same
constructor arguments, parameter names and defaults, methods, units and array shapes, for the
part of its interface that scripts and Prospector use. This page lists what is covered and
every known difference.

## Covered

| python-fsps | opensps | Notes |
|---|---|---|
| `StellarPopulation(compute_vega_mags, vactoair_flag, zcontinuous, **params)` | same | plus `libraries=`, `data_home=` |
| `params[...]`, `params.all_params`, `ssp_params`, `csp_params`, `dirty`, `dirtiness` | same | `extra_params`: parameters of libopensps that python-fsps lacks |
| `get_spectrum(zmet, tage, peraa)` | same | L_sun/Hz, or L_sun/A with `peraa` (factor 3e18 / lambda^2, as python-fsps) |
| `get_mags(zmet, tage, redshift, bands)` | same | see `redshift` below |
| `set_tabular_sfh(age, sfr, Z)` | same | ages in Gyr, SFR in M_sun/yr |
| `filter_data()` | same | |
| `wavelengths`, `emline_wavelengths`, `zlegend`, `solar_metallicity`, `ssp_ages` | same | |
| `log_age`, `stellar_mass`, `formed_mass`, `log_lbol`, `sfr`, `dust_mass`, `emline_luminosity` | same | scalars / 1-D for `tage > 0`, arrays over ages for `tage = 0` |
| `libraries`, `isoc_library`, `spec_library`, `duste_library` | see below | |
| `fsps.list_filters()`, `find_filter()`, `get_filter()` | `opensps.list_filters()`, ... | `Filter.index`, `.name`, `.fullname` |

Prospector's `SSPBasis`, `CSPSpecBasis` and `FastStepBasis` run unchanged after
`opensps.compat.fsps.install_as_fsps()` (tested: `tests/test_prospector.py`).

## Differences

**Physics.** The numbers are those of libopensps, which differs from the legacy FSPS inside
python-fsps where legacy is grid-dependent, loses energy or uses rounded constants. The largest:
legacy's `gsig4pi = 1/4.13e10` makes every main-library spectrum 1.77% brighter than the exact
1/(4 pi sigma); dust self-absorption summed analytically (up to 1.4% of the dust emission for
`dust_type = 0`); exact IMF integrals; velocity smoothing exact across grid-spacing jumps. The
full list, with causes and sizes, is in the core repository, `tools/legacy_parity/README.md`.
`zcontinuous = 1` interpolates the SSPs linearly in log Z as the core does.

**Libraries per instance.** python-fsps fixes the isochrone, spectral and dust-emission
libraries when it is compiled; here `StellarPopulation(libraries=("mist", "miles", "DL07"))`
chooses them at run time. `libraries` reports what was requested (`None` for "the core's
default"), not names read back from the library.

**One context per instance.** python-fsps keeps one global state shared by all instances (and
asserts that `compute_vega_mags` and `vactoair_flag` agree between them); each opensps instance
has its own, so instances with different settings coexist and can run in different threads.

**`get_mags(redshift=z)`** with `z` different from `params["zred"]` recomputes the population at
`z` (IGM included) and restores `zred`; python-fsps re-observes the stored spectra at `z`
without IGM.

**Errors.** Invalid parameters and inputs raise `ValueError` / `KeyError` /
`opensps.FSPSError` where python-fsps uses `assert` (e.g. in `check_params` and
`set_tabular_sfh`). A tabular SFH must be set before computing with `sfh = 3`.

**Tabular SFH mass.** For `sfh = 3` at `tage > 0` the masses and spectra are absolute (M_sun
formed from the SFR table, as in python-fsps), but `formed_mass` is reported as 1 by the core
(an open issue of the core). Prospector does not use `formed_mass` (it divides by its own
total mass).

**Size-1 arrays and NumPy scalars** are accepted as parameter values (as python-fsps, through
f2py, does); larger arrays raise `ValueError`.

**Extras.** `context` (the underlying `opensps.Context`, e.g. for batch calls with the same
libraries), `emline_names`, and the parameters `add_agn_dust`, `add_xrb_emission`, `frac_xrb`,
`dust3`, `smoothspec_fast`, `use_isoc_mdot`, `interpolation_type`, `pzcon`, `tiny_logt`,
`nebular_smooth_init`, `setup_nebular_gaussians`, `om0`, `ol0`, `H0`, `compute_mags`,
`compute_indices`.

## Not provided

`sfr_avg`, `isochrones`, `smoothspec`, `set_lsf`, `_ztinterp`, `_all_ssp_spec`,
`_get_stellar_spectrum`, `_csp_young_old` (used by Prospector's AGN basis), `resolutions`, the
special value `tage = -99`. These will not be added to this layer; where the functionality
matters, it belongs in the native API.

There is no automated comparison with python-fsps: its numbers come from legacy FSPS, whose
differences from libopensps are documented case by case in the core repository
(`tools/legacy_parity/README.md`).

## Porting to the native API

| python-fsps | opensps |
|---|---|
| `sp = fsps.StellarPopulation(zcontinuous=1, **p)` | `ctx = opensps.Context("mist", "miles", "DL07"); ctx.set_params(p)` |
| `sp.params["dust2"] = 0.3` | `ctx.set("dust2", 0.3)` |
| `sp.get_spectrum(tage=5.0)` | `ctx.set("tage", 5.0); out = ctx.compute_csp(1); out["spec"][0]` |
| `sp.get_mags(tage=5.0, bands=b)` | `out["mags"][0][[opensps.get_filter(x).index for x in b]]` |
| `sp.stellar_mass`, `sp.log_lbol`, `sp.emline_luminosity` | `out["mass"]`, `out["lbol"]`, `out["emlines"]` |
| `sp.set_tabular_sfh(t, sfr)` | `ctx.set_tabular_sfh(t, sfr)` |
| `sp.wavelengths`, `sp.ssp_ages` | `ctx.wavelengths`, `ctx.log_ages` |
| a loop over parameter values | `ctx.compute_csp_batch({"dust2": array, ...})` |

