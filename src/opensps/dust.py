"""Applying the batch's dust model to its intrinsic young and old parts.

``Context.compute_csp_batch(..., parts=True)`` returns the young (ages <= ``dust_tesc``) and old
spectra as the attenuation stage receives them. :func:`attenuate_parts` applies that stage, so a
caller can vary the dust after computing the stellar populations once (e.g. emulator training
sets). It reproduces the batch's spectrum exactly when dust emission, the AGN torus, IGM
absorption and smoothing are off; those act on the attenuated total.
"""

import numpy as np

__all__ = ["attenuate_parts", "power_law_curve", "V_BAND"]

#: Reference wavelength of the attenuation curves and of dust1_index [A]
V_BAND = 5500.0


def power_law_curve(wavelengths, dust_index):
    """The ``dust_type = 0`` curve, (lambda / 5500 A)^dust_index; shape (n, nspec) for a vector
    of indices."""
    lam = np.asarray(wavelengths, dtype=float)
    idx = np.asarray(dust_index, dtype=float)
    return (lam / V_BAND) ** idx[..., None] if idx.ndim else (lam / V_BAND) ** idx


def _col(x):
    x = np.asarray(x, dtype=float)
    return x[..., None] if x.ndim else x


def attenuate_parts(wavelengths, young, old, dust1=0.0, dust2=0.0, dust1_index=-1.0, dust3=0.0,
                    frac_obrun=0.0, frac_nodust=0.0, curve=None, dust_index=-0.7):
    """The attenuated spectrum from the intrinsic parts (the batch's dust stage):

        (young (T_birth (1 - frac_obrun) + frac_obrun) + old T_dust3)
            (T_diffuse (1 - frac_nodust) + frac_nodust)

    with ``T = exp(-tau)``, ``tau_birth = dust1 (lambda / 5500 A)^dust1_index``,
    ``tau_diffuse = dust2 curve``, ``tau_dust3 = dust3 curve``.

    :param wavelengths: (nspec,) [A], e.g. ``Context.wavelengths``.
    :param young, old: (n, nspec) or (nspec,) parts.
    :param dust1, dust2, dust1_index, dust3, frac_obrun, frac_nodust: scalars or (n,) arrays.
    :param curve: the diffuse attenuation curve per pixel, (nspec,) or (n, nspec), for the
        context's ``dust_type``; default the ``dust_type = 0`` power law with ``dust_index``.
        (For ``dust_type = 3`` the curve is the optical depth itself and ``dust2`` must be 1.)
    """
    lam = np.asarray(wavelengths, dtype=float)
    if curve is None:
        curve = power_law_curve(lam, dust_index)
    curve = np.asarray(curve, dtype=float)
    d1 = _col(dust1)
    tb = np.where(d1 > 0.0, np.exp(-d1 * (lam / V_BAND) ** _col(dust1_index)), 1.0)
    d3 = _col(dust3)
    to = np.where(d3 != 0.0, np.exp(-d3 * curve), 1.0)
    td = np.exp(-_col(dust2) * curve)
    fob, fnd = _col(frac_obrun), _col(frac_nodust)
    return (young * (tb * (1.0 - fob) + fob) + old * to) * (td * (1.0 - fnd) + fnd)
