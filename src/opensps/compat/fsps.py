"""A python-fsps compatible ``StellarPopulation`` on top of :class:`opensps.Context`.

A migration aid, frozen: it covers what existing scripts and Prospector use and gets no new
features for python-fsps parity. New code should use :class:`opensps.Context` directly.

Covers the part of python-fsps that scripts and Prospector (``SSPBasis``, ``CSPSpecBasis``,
``FastStepBasis``) use: the ``params`` mapping, ``get_spectrum``, ``get_mags``,
``set_tabular_sfh``, the grids (``wavelengths``, ``ssp_ages``, ``zlegend``, ...) and the
statistics of the last model (``stellar_mass``, ``formed_mass``, ``log_lbol``, ``sfr``,
``dust_mass``, ``emline_luminosity``, ``log_age``). Units and array shapes follow python-fsps.
Differences are listed in docs/migrating-from-python-fsps.md.

Each ``StellarPopulation`` owns its own library context (python-fsps shares one global state
between all instances), so several populations with different settings can coexist.
"""

import sys
import types

import numpy as np

from .. import filters as _filters
from .._core import Context, FSPSError

__all__ = ["StellarPopulation", "ParameterSet", "install_as_fsps"]

#: The L_nu -> L_lambda factor of python-fsps (c in A/s, rounded as there)
_LIGHTSPEED_PERAA = 3e18


class ParameterSet:
    """Dict-like parameter access; setting a value passes it to the library at once and marks
    the population for recomputation if it changed."""

    #: Parameters that change the SSPs (as in python-fsps)
    ssp_params = [
        "imf_type", "imf_upper_limit", "imf_lower_limit", "imf1", "imf2", "imf3", "vdmc",
        "mdave", "dell", "delt", "sbss", "fbhb", "pagb", "add_stellar_remnants",
        "tpagb_norm_type", "add_agb_dust_model", "agb_dust", "redgb", "agb", "masscut", "fcstar",
        "evtype", "use_wr_spectra", "logt_wmb_hot", "smooth_lsf",
    ]

    #: Parameters that act on the CSP (as in python-fsps)
    csp_params = [
        "smooth_velocity", "redshift_colors", "compute_light_ages", "nebemlineinspec",
        "dust_type", "add_dust_emission", "add_neb_emission", "add_neb_continuum", "cloudy_dust",
        "add_igm_absorption", "zmet", "sfh", "wgp1", "wgp2", "wgp3", "tau", "const", "tage",
        "fburst", "tburst", "dust1", "dust2", "logzsol", "zred", "pmetals", "dust_clumps",
        "frac_nodust", "dust_index", "dust_tesc", "frac_obrun", "uvb", "mwr", "dust1_index",
        "sf_start", "sf_trunc", "sf_slope", "duste_gamma", "duste_umin", "duste_qpah",
        "sigma_smooth", "min_wave_smooth", "max_wave_smooth", "gas_logu", "gas_logz",
        "igm_factor", "fagn", "agn_tau",
    ]

    #: Parameters of this library that python-fsps does not have
    extra_params = [
        "add_agn_dust", "add_xrb_emission", "frac_xrb", "dust3", "smoothspec_fast",
        "use_isoc_mdot", "interpolation_type", "pzcon", "tiny_logt", "nebular_smooth_init",
        "setup_nebular_gaussians", "om0", "ol0", "H0", "compute_mags", "compute_indices",
    ]

    def __init__(self, owner, values):
        self._owner = owner
        self._params = dict(values)
        self.dirtiness = 2

    @property
    def all_params(self):
        return self.ssp_params + self.csp_params + self.extra_params

    @property
    def dirty(self):
        return self.dirtiness > 0

    def __getitem__(self, k):
        return self._params[k]

    def __setitem__(self, k, v):
        if k not in self._params:
            raise KeyError(f"unknown parameter {k!r}")
        old = self._params[k]
        changed = np.any(np.asarray(old) != np.asarray(v))
        if changed:
            self._owner._ctx.set(k, v)
            self._params[k] = v
            self.dirtiness = max(2 if k in self.ssp_params else 1, self.dirtiness)

    def __contains__(self, k):
        return k in self._params

    def __iter__(self):
        return iter(self._params)

    def __len__(self):
        return len(self._params)

    def keys(self):
        return self._params.keys()

    def items(self):
        return self._params.items()

    iteritems = items

    def get(self, k, default=None):
        return self._params.get(k, default)

    def update(self, other=(), **kwargs):
        for k, v in dict(other, **kwargs).items():
            self[k] = v

    def check_params(self):
        """The checks python-fsps makes before a computation (ValueError instead of assert)."""
        p = self._params
        nz = self._owner._ctx.nz
        if not 1 <= int(p["zmet"]) <= nz:
            raise ValueError(f"zmet={p['zmet']} out of range [1, {nz}]")
        if int(p["dust_type"]) not in range(7):
            raise ValueError(f"dust_type={p['dust_type']} out of range [0, 6]")
        if int(p["imf_type"]) not in range(6):
            raise ValueError(f"imf_type={p['imf_type']} out of range [0, 5]")
        if not (p["tage"] <= 0 or p["tage"] > p["sf_start"]):
            raise ValueError(f"sf_start={p['sf_start']} is greater than tage={p['tage']}")
        if p["const"] + p["fburst"] > 1:
            raise ValueError("const + fburst > 1")

    def __repr__(self):
        return f"ParameterSet({self._params!r})"


#: python-fsps defaults (and this library's own for the extra parameters)
DEFAULTS = dict(
    add_agb_dust_model=True, add_dust_emission=True, add_igm_absorption=False,
    add_neb_emission=False, add_neb_continuum=True, add_stellar_remnants=True,
    redshift_colors=False, compute_light_ages=False, nebemlineinspec=True, smooth_velocity=True,
    smooth_lsf=False, cloudy_dust=False, agb_dust=1.0, tpagb_norm_type=2, dell=0.0, delt=0.0,
    redgb=1.0, agb=1.0, fcstar=1.0, fbhb=0.0, sbss=0.0, pagb=1.0, zred=0.0, zmet=1,
    logzsol=0.0, pmetals=2.0, imf_type=2, imf_upper_limit=120, imf_lower_limit=0.08, imf1=1.3,
    imf2=2.3, imf3=2.3, vdmc=0.08, mdave=0.5, evtype=-1, use_wr_spectra=1, logt_wmb_hot=0.0,
    masscut=150.0, sigma_smooth=0.0, min_wave_smooth=1e3, max_wave_smooth=1e4, gas_logu=-2,
    gas_logz=0.0, igm_factor=1.0, sfh=0, tau=1.0, const=0.0, sf_start=0.0, sf_trunc=0.0,
    tage=0.0, dust_tesc=7.0, fburst=0.0, tburst=11.0, sf_slope=0.0, dust_type=0, dust1=0.0,
    dust2=0.0, dust_clumps=-99.0, frac_nodust=0.0, frac_obrun=0.0, dust_index=-0.7,
    dust1_index=-1.0, mwr=3.1, uvb=1.0, wgp1=1, wgp2=1, wgp3=1, duste_gamma=0.01,
    duste_umin=1.0, duste_qpah=3.5, fagn=0.0, agn_tau=10.0,
    # this library only
    add_agn_dust=1, add_xrb_emission=0, frac_xrb=1.0, dust3=0.0, smoothspec_fast=1,
    use_isoc_mdot=0, interpolation_type=0, pzcon=0, tiny_logt=0.0, nebular_smooth_init=100.0,
    setup_nebular_gaussians=0, om0=0.27, ol0=0.73, H0=72.0, compute_mags=1, compute_indices=1,
)


class StellarPopulation:
    """python-fsps compatible stellar population.

    :param compute_vega_mags: magnitudes in the Vega system (default AB).
    :param vactoair_flag: wavelengths in air (default vacuum).
    :param zcontinuous: 0 SSPs at ``zmet``; 1 interpolated at ``logzsol``; 2 a metallicity
        distribution with ``pmetals``; 3 tabular SFH metallicities.
    :param libraries: ``(isochrones, spectra, dust emission)`` names, e.g.
        ``("mist", "miles", "DL07")``; python-fsps fixes them at compile time, here they are
        chosen per instance. ``None`` (default) uses the library defaults.
    :param data_home: data root for this instance (see :func:`opensps.set_data_home`).
    :param kwargs: initial parameter values (python-fsps names, plus this library's extras).
    """

    def __init__(self, compute_vega_mags=False, vactoair_flag=False, zcontinuous=0,
                 libraries=None, data_home=None, **kwargs):
        unknown = set(kwargs) - set(DEFAULTS)
        if unknown:
            raise TypeError(f"__init__() got an unexpected keyword argument '{sorted(unknown)[0]}'")
        if zcontinuous not in (0, 1, 2, 3):
            raise ValueError("zcontinuous must be 0, 1, 2 or 3")
        isoc, spec, dust = (tuple(libraries) + (None, None, None))[:3] if libraries else (None,) * 3
        self._libraries = (isoc, spec, dust)
        self._zcontinuous = zcontinuous
        values = dict(DEFAULTS)
        values.update(kwargs)
        early = {"compute_vega_mags": int(bool(compute_vega_mags)),
                 "vactoair_flag": int(bool(vactoair_flag))}
        self._ctx = Context(isoc, spec, dust, zin=-1, params=early, data_home=data_home)
        self._data_home = self._ctx.paths[0] or None
        for k, v in values.items():
            self._ctx.set(k, v)
        self.params = ParameterSet(self, values)
        self._out = None
        self._has_table = False

    # ------------------------------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------------------------------

    @property
    def context(self):
        """The underlying :class:`opensps.Context` (e.g. for batch calls with the same
        libraries and settings)."""
        return self._ctx

    def _compute_csp(self):
        self.params.check_params()
        if self.params["sfh"] == 3 and not self._has_table:
            raise ValueError("sfh=3 needs a tabular SFH: call set_tabular_sfh() first")
        self._out = self._ctx.compute_csp(self._zcontinuous)
        self.params.dirtiness = 0

    def _ensure(self):
        if self.params.dirty or self._out is None:
            self._compute_csp()
        return self._out

    def _prepare(self, zmet, tage):
        self.params["tage"] = tage
        if zmet is not None:
            self.params["zmet"] = zmet

    def get_spectrum(self, zmet=None, tage=0.0, peraa=False):
        """Spectra of the current population: ``(wavelengths, spectrum)``.

        :param tage: age in Gyr; 0 (default) for all SSP ages (shape (ntfull, nspec)).
        :param peraa: L_sun/A instead of L_sun/Hz.
        """
        self._prepare(zmet, tage)
        out = self._ensure()
        wave = self.wavelengths
        spec = out["spec"]
        if peraa:
            spec = spec * (_LIGHTSPEED_PERAA / wave**2)[None, :]
        else:
            spec = spec.copy()
        if tage > 0.0:
            return wave, spec[0]
        return wave, spec

    def get_mags(self, zmet=None, tage=0.0, redshift=None, bands=None):
        """Magnitudes (AB, or Vega with ``compute_vega_mags``), shape (nbands,) for
        ``tage > 0`` else (ntfull, nbands); ``bands`` selects and orders filters by name.

        ``redshift`` different from ``params['zred']`` recomputes the population at that
        redshift (python-fsps instead re-observes the stored spectra, without IGM).
        """
        self._prepare(zmet, tage)
        zred = self.params["zred"]
        if redshift is not None and redshift != zred:
            self.params["zred"] = redshift
            try:
                out = self._ensure()
            finally:
                self.params["zred"] = zred
        else:
            out = self._ensure()
        if "mags" not in out:
            raise FSPSError(309, "magnitudes not computed (compute_mags = 0 or fast mode)")
        mags = out["mags"]
        if bands is not None:
            mags = mags[:, _filters.band_indices(bands, self._data_home)]
        else:
            mags = mags.copy()
        return mags[0] if tage > 0.0 else mags

    def set_tabular_sfh(self, age, sfr, Z=None):
        """Tabular SFH for ``sfh=3``: ``age`` [Gyr, increasing], ``sfr`` [M_sun/yr], optional
        metallicities ``Z`` (absolute, for ``zcontinuous=3``)."""
        age = np.asarray(age, dtype=float)
        sfr = np.asarray(sfr, dtype=float)
        if age.shape != sfr.shape:
            raise ValueError("age and sfr have different size.")
        if np.any(age[1:] <= age[:-1]):
            raise ValueError("Ages must be increasing.")
        if not np.any(sfr > 1e-33):
            raise ValueError("At least one sfr must be > 1e-33.")
        if np.any(sfr < 0):
            raise ValueError("sfr cannot be negative.")
        if Z is None:
            if self._zcontinuous == 3:
                raise ValueError("zcontinuous=3 needs metallicities Z")
        else:
            Z = np.asarray(Z, dtype=float)
            if Z.shape != age.shape:
                raise ValueError("Z and age have different size.")
            if np.any(Z < 0):
                raise ValueError("All values of Z must be greater than or equal 0.")
            if self._zcontinuous != 3:
                raise ValueError("_zcontinuous must be 3 for multi-Z tabular.")
            if self.params["add_neb_emission"]:
                raise ValueError("Cannot compute nebular emission with multi-metallicity tabular.")
        self._ctx.set_tabular_sfh(age, sfr, Z)
        self._has_table = True
        if self.params["sfh"] == 3:
            self.params.dirtiness = max(1, self.params.dirtiness)
        else:
            print("Warning: You are setting a tabular SFH, but the ``sfh`` parameter is not 3")

    def filter_data(self):
        """(effective wavelengths [A], AB magnitude of Vega, absolute AB magnitude of the Sun)."""
        return self._ctx.filter_data()

    # ------------------------------------------------------------------------------------------
    # Grids
    # ------------------------------------------------------------------------------------------

    @property
    def wavelengths(self):
        """Wavelengths of the spectra [A]."""
        return np.array(self._ctx.wavelengths)

    @property
    def emline_wavelengths(self):
        """Emission-line wavelengths [A]."""
        return np.array(self._ctx.emline_wavelengths)

    @property
    def emline_names(self):
        """Emission-line names (not in python-fsps)."""
        return _filters.emline_names(self._data_home)

    @property
    def zlegend(self):
        """Metallicities Z of the isochrone grid."""
        return np.array(self._ctx.zlegend)

    @property
    def solar_metallicity(self):
        return self._ctx.zsol

    @property
    def ssp_ages(self):
        """SSP ages, log10(age / yr)."""
        return np.array(self._ctx.log_ages)

    @property
    def libraries(self):
        """(isochrones, spectra, dust emission) as requested at construction (``None`` for a
        library default)."""
        return self._libraries

    @property
    def isoc_library(self):
        return self._libraries[0]

    @property
    def spec_library(self):
        return self._libraries[1]

    @property
    def duste_library(self):
        return self._libraries[2]

    # ------------------------------------------------------------------------------------------
    # Statistics of the last population
    # ------------------------------------------------------------------------------------------

    def _stat(self, key):
        out = self._ensure()
        v = out[key]
        if self.params["tage"] > 0:
            return v[0]
        return v.copy()

    @property
    def log_age(self):
        """log10(age / yr)."""
        return self._stat("age")

    @property
    def stellar_mass(self):
        """Surviving stellar mass [M_sun] (with remnants if ``add_stellar_remnants``)."""
        return self._stat("mass")

    @property
    def log_lbol(self):
        """log10(L_bol / L_sun)."""
        return self._stat("lbol")

    @property
    def sfr(self):
        """Star formation rate [M_sun/yr]."""
        return self._stat("sfr")

    @property
    def dust_mass(self):
        """Dust mass [M_sun]."""
        return self._stat("mdust")

    @property
    def formed_mass(self):
        """Mass formed [M_sun]."""
        return self._stat("mformed")

    @property
    def emline_luminosity(self):
        """Emission-line luminosities [L_sun], shape (nemline,) for ``tage > 0``."""
        return self._stat("emlines")


def install_as_fsps():
    """Make ``import fsps`` return a module backed by this package (``StellarPopulation``,
    ``list_filters``, ``find_filter``, ``get_filter``), so that code written for python-fsps
    (e.g. Prospector's ``SSPBasis``) runs on it. Call before that code imports ``fsps``."""
    from .. import __version__

    mod = types.ModuleType("fsps")
    mod.StellarPopulation = StellarPopulation
    mod.list_filters = _filters.list_filters
    mod.find_filter = _filters.find_filter
    mod.get_filter = _filters.get_filter
    mod.__version__ = __version__
    mod.__opensps__ = True
    sys.modules["fsps"] = mod
    return mod
