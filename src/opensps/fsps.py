"""Drop-in compatibility layer with python-fsps (StellarPopulation)."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from opensps.context import Context


_FSPS_LOCAL_PARAMS = {"zcontinuous", "compute_vega_mags", "vactoair_flag", "custom_filter_key"}


class _ParamDict(dict):
    """Dictionary that automatically pushes parameter changes down to the OpenSPS context."""

    def __init__(self, context: Context, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._context = context

    def __setitem__(self, key: str, value: Any):
        super().__setitem__(key, value)
        if key not in _FSPS_LOCAL_PARAMS:
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
            batch = self._ctx.batch({"tage": tage})
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
            batch = self._ctx.batch({"tage": tage})
            res = batch.compute(compute_spec=False)
            mags = res.mags[0]
        else:
            batch = self._ctx.batch({"tage": 10.0})
            res = batch.compute(compute_spec=False)
            mags = res.mags[0]

        if bands is not None:
            return mags[bands]
        return mags
