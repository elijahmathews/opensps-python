"""Drop-in compatibility layer with python-fsps (StellarPopulation)."""

from __future__ import annotations

import weakref

import numpy as np
from typing import Dict, List, Optional, Sequence, Tuple, Union, Any
from opensps.constants import PARAM_TO_BGRAD
from opensps.context import Context

# Parameters a batch takes per model (the gradient parameters, plus the SFH ones that are not
# differentiated). A python-fsps style call sets them on the object; they must reach the batch,
# because a batch that is not given them uses the context's single-metallicity SSP grid and the
# context's SFH only (``logzsol`` set on the context does not change a batch).
_BATCH_PARAMS = frozenset(PARAM_TO_BGRAD) | {"tage", "tburst"}


_FSPS_LOCAL_PARAMS = {"zcontinuous", "compute_vega_mags", "vactoair_flag", "custom_filter_key"}

# The names of python-fsps' ``StellarPopulation.params.all_params``: code written for python-fsps
# (Prospector's CSPSpecBasis) asks ``name in ssp.params.all_params`` to decide which of its own
# parameters to pass on. A name the libopensps build does not support still raises when it is set.
FSPS_ALL_PARAMS = (
    "add_agb_dust_model", "add_dust_emission", "add_igm_absorption", "add_neb_continuum",
    "add_neb_emission", "add_stellar_remnants", "add_xrb", "agb_dust", "agn_tau", "cloudy_dust",
    "compute_light_ages", "const", "dell", "delt", "dust1", "dust1_index", "dust2", "dust_clumps",
    "dust_index", "dust_tesc", "dust_type", "duste_gamma", "duste_qpah", "duste_umin", "fagn",
    "fbhb", "fburst", "frac_nodust", "frac_obrun", "gas_logu", "gas_logz", "igm_factor", "imf1",
    "imf2", "imf3", "imf_lower_limit", "imf_type", "imf_upper_limit", "logzsol", "masscut", "max_wave_smooth",
    "mdave", "min_wave_smooth", "mwr", "nebemlineinspec", "pagb", "pmetals", "redgb", "sbss",
    "sf_slope", "sf_start", "sf_trunc", "sfh", "sigma_smooth", "smooth_lsf", "smooth_velocity",
    "tage", "tau", "tburst", "tpagb_norm_type", "use_wr_spectra", "uvb", "vdmc", "wgp1", "wgp2",
    "wgp3", "zred", "zmet", "compute_vega_mags", "vactoair_flag", "zcontinuous",
)


_UNSET = object()

# Pickling a StellarPopulation (dynesty and multiprocessing pools pickle the likelihood, which holds
# the SPS object) cannot rebuild the context per task: it is a handle to library state. Within a
# process tree started by ``fork`` after the object exists, every worker already has its copy: the
# pickle is a key into this registry, resolved to that copy.
_INSTANCES = weakref.WeakValueDictionary()


def _resolve_instance(key: int):
    try:
        return _INSTANCES[key]
    except KeyError:
        raise RuntimeError("an opensps.StellarPopulation can be unpickled only in a process forked after it "
                           "was created (multiprocessing 'fork' pools); create the pool after the object") from None


def _same_value(a: Any, b: Any) -> bool:
    try:
        return bool(np.array_equal(a, b))
    except Exception:
        return False


class _ParamDict(dict):
    """Dictionary that automatically pushes parameter changes down to the OpenSPS context."""

    def __init__(self, context: Context, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._context = context

    @property
    def all_params(self) -> Tuple[str, ...]:
        """Names of the FSPS parameters (as python-fsps' ``params.all_params``)."""
        return FSPS_ALL_PARAMS

    def __setitem__(self, key: str, value: Any):
        old = self.get(key, _UNSET)
        super().__setitem__(key, value)
        if key in _FSPS_LOCAL_PARAMS:
            return
        # Prospector sets every parameter on every call. A set that changes nothing is skipped:
        # re-setting some parameters (the IMF, say) invalidates the context's SSP basis, which then
        # is rebuilt (10x the cost of a single-model call).
        if old is not _UNSET and _same_value(old, value):
            return
        self._context.set_param(key, value)


class StellarPopulation:
    """Drop-in compatible replacement for `fsps.StellarPopulation`.

    Provides the same API surface used by Prospector and python-fsps scripts,
    backed by high-performance libopensps contexts.
    """

    def __init__(
        self,
        compute_vega_mags: bool = False,
        vactoair_flag: bool = False,
        zcontinuous: int = 1,
        custom_filter_key: Optional[str] = None,
        **kwargs,
    ):
        # Extract library overrides if present
        zmet = kwargs.pop("zmet", -1)
        isoc = kwargs.pop("isoc_type", kwargs.pop("isoc", "mist"))
        spec = kwargs.pop("spec_type", kwargs.pop("spec", "miles"))
        dust = kwargs.pop("dust_type", kwargs.pop("dust", "DL07"))

        self._ctx = Context(zmet=zmet, isoc=isoc, spec=spec, dust=dust)
        _INSTANCES[id(self)] = self
        self.params = _ParamDict(self._ctx)
        self._stellar_mass = 0.0
        self._dust_mass = 0.0
        self._log_lbol = 0.0
        self._sfr = 0.0
        self._emline_luminosity = None

        # Configure initial flags
        self.params["compute_vega_mags"] = 1 if compute_vega_mags else 0
        self.params["vactoair_flag"] = 1 if vactoair_flag else 0
        self.params["zcontinuous"] = zcontinuous

        # Apply any additional initial parameters
        for k, v in kwargs.items():
            self.params[k] = v

    def __reduce__(self):
        return (_resolve_instance, (id(self),))

    def _batch_params(self, tage: float) -> Dict[str, float]:
        """The per-model parameters of a one-model batch: this object's, at ``tage`` (Gyr)."""
        p = {}
        for k, v in self.params.items():
            if k in _BATCH_PARAMS and v is not None:
                try:
                    p[k] = float(np.squeeze(v))
                except (TypeError, ValueError):
                    pass
        if int(self.params.get("zcontinuous", 1)) == 0:
            p.pop("logzsol", None)           # SSPs at the context's zmet grid point
        p["tage"] = float(tage)
        return p

    @property
    def context(self) -> Context:
        """The underlying OpenSPS context."""
        return self._ctx

    @property
    def wavelengths(self) -> np.ndarray:
        """Rest-frame wavelength grid [A]."""
        return self._ctx.wavelengths

    @property
    def ssp_ages(self) -> np.ndarray:
        """SSP log10(age/yr) grid."""
        return self._ctx.log_ages

    @property
    def zlegend(self) -> np.ndarray:
        """Metallicity grid in mass fraction Z."""
        return self._ctx.zlegend

    @property
    def emline_wavelengths(self) -> np.ndarray:
        """Emission line wavelengths [A]."""
        return self._ctx.emlambda

    @property
    def libraries(self) -> Tuple[str, str, str]:
        """Tuple of (isoc_type, spec_type, dust_type)."""
        return ("mist", "miles", "DL07")

    @property
    def solar_metallicity(self) -> float:
        """Solar metallicity Z_sun in mass fraction."""
        return self._ctx.zsol

    @property
    def stellar_mass(self) -> float:
        """Surviving stellar mass in solar masses."""
        return self._stellar_mass

    @property
    def dust_mass(self) -> float:
        """Dust mass in solar masses."""
        return self._dust_mass

    @property
    def log_lbol(self) -> float:
        """Log10 bolometric luminosity in L_sun."""
        return self._log_lbol

    @property
    def sfr(self) -> float:
        """Star formation rate in M_sun / yr."""
        return self._sfr

    @property
    def emline_luminosity(self) -> np.ndarray:
        """Nebular emission line luminosities in L_sun."""
        if self._emline_luminosity is None:
            return np.zeros(self._ctx.nemline)
        return self._emline_luminosity

    def get_spectrum(
        self,
        tage: float = 0.0,
        peraa: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute and return the spectrum of the stellar population.

        Parameters
        ----------
        tage : float, optional
            Age of the population in Gyr. If tage <= 0, computes the full age grid.
        peraa : bool, optional
            If True, returns flux in L_sun / A. If False, returns flux in L_sun / Hz.

        Returns
        -------
        wavelengths : ndarray
            Wavelength array in Angstroms.
        spectrum : ndarray
            Flux array. Shape (nspec,) if tage > 0, else (ntfull, nspec).
        """
        if tage > 0:
            self.params["tage"] = tage
            batch = self._ctx.batch(self._batch_params(tage))
            res = batch.compute(compute_spec=True, compute_emlines=True)
            self._stellar_mass = float(res.mass[0])
            self._dust_mass = float(res.mdust[0])
            self._log_lbol = float(res.lbol[0])
            self._sfr = float(res.sfr[0])
            self._emline_luminosity = res.emlines[0]
            spec = res.spec[0]
            if peraa:
                c_light = 2.99792458e18  # A / s
                spec = spec * (c_light / (self.wavelengths ** 2))
            return self.wavelengths, spec
        else:
            # Single SSP grid
            spec, _, _ = self._ctx.compute_ssp()
            if peraa:
                c_light = 2.99792458e18
                spec = spec * (c_light / (self.wavelengths[None, :] ** 2))
            return self.wavelengths, spec

    def get_mags(
        self,
        tage: float = 0.0,
        bands: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """Compute and return magnitudes for the stellar population.

        Parameters
        ----------
        tage : float, optional
            Age of the population in Gyr.
        bands : sequence of str, optional
            Band names to retrieve (if None, returns all nbands).

        Returns
        -------
        mags : ndarray of shape (nbands,)
        """
        if tage > 0:
            self.params["tage"] = tage
            batch = self._ctx.batch(self._batch_params(tage))
            res = batch.compute(compute_spec=False)
            mags = res.mags[0]
        else:
            batch = self._ctx.batch(self._batch_params(10.0))
            res = batch.compute(compute_spec=False)
            mags = res.mags[0]

        if bands is not None:
            return mags[bands]
        return mags
