"""Star formation histories as SSP weights, for batches of non-parametric SFHs.

``CSPBatch`` takes a matrix of SSP weights per model; these helpers build it from piecewise-linear
SFHs (a step function with short ramps is a piecewise-linear SFH), vectorised over models, without
the per-model tabular path of the context.  The weights are those of libopensps: the SSP spectra are
interpolated linearly in log10(age) between the grid ages, so the weight of a grid age is the integral
of the SFR times the hat function centred on it; the youngest SSP carries everything younger than its
age.  They sum to the mass formed.
"""

from __future__ import annotations

import numpy as np

__all__ = ["step_table", "ssp_weights_from_table"]


def step_table(edges: np.ndarray, sfr: np.ndarray, ramp: float = 1.0e5):
    """Piecewise-linear table of a step-function SFH.

    Parameters
    ----------
    edges : array_like, shape (N, B + 1)
        Bin edges in lookback time [yr], ascending, ``edges[:, 0] == 0``.
    sfr : array_like, shape (N, B)
        Star formation rate of each bin [M_sun/yr], bin 0 the youngest.
    ramp : float
        Half width [yr] of the linear ramp centred on every interior edge. A tabular SFH is linear
        between its points, so a step has to be a ramp; 1e5 yr is far below the SSP age grid's
        resolution at the ages where the bins meet.

    Returns
    -------
    t, s : ndarray, shape (N, 2 B)
        Lookback times [yr] (ascending) and SFRs [M_sun/yr] of the table points.
    """
    edges = np.atleast_2d(np.asarray(edges, dtype=np.float64))
    sfr = np.atleast_2d(np.asarray(sfr, dtype=np.float64))
    n, nb = sfr.shape
    if edges.shape != (n, nb + 1):
        raise ValueError(f"edges must have shape {(n, nb + 1)}, got {edges.shape}")
    t = np.empty((n, 2 * nb))
    s = np.empty((n, 2 * nb))
    t[:, 0], s[:, 0] = 0.0, sfr[:, 0]
    for k in range(1, nb):
        t[:, 2 * k - 1], s[:, 2 * k - 1] = edges[:, k] - ramp, sfr[:, k - 1]
        t[:, 2 * k], s[:, 2 * k] = edges[:, k] + ramp, sfr[:, k]
    t[:, -1], s[:, -1] = edges[:, -1], sfr[:, -1]
    return t, s


def _xlogx(x):
    return np.where(x > 0, x * np.log(np.where(x > 0, x, 1.0)), 0.0)


def _cumulative_moments(t, s, upto):
    """int_0^T s dtau and int_0^T s log10(tau) dtau for piecewise-linear s; ``upto`` (N, K)."""
    tk, tk1 = t[:, :-1], t[:, 1:]
    sk, sk1 = s[:, :-1], s[:, 1:]
    slope = (sk1 - sk) / (tk1 - tk)
    icpt = sk - slope * tk
    x1 = np.clip(upto[:, None, :], tk[:, :, None], tk1[:, :, None])
    a, b, lo = icpt[:, :, None], slope[:, :, None], tk[:, :, None]

    def antiderivative(tau):   # int (a + b tau) ln tau dtau
        return a * (_xlogx(tau) - tau) + b * (0.5 * tau * _xlogx(tau) - 0.25 * tau * tau)

    m0 = (a * (x1 - lo) + 0.5 * b * (x1 * x1 - lo * lo)).sum(axis=1)
    m1 = ((antiderivative(x1) - antiderivative(lo)) / np.log(10.0)).sum(axis=1)
    return m0, m1


def ssp_weights_from_table(log_ages: np.ndarray, t: np.ndarray, s: np.ndarray) -> np.ndarray:
    """SSP weights (N, ntfull) of piecewise-linear SFHs.

    Parameters
    ----------
    log_ages : array_like, shape (ntfull,)
        The context's SSP grid, log10(age/yr) (``Context.log_ages``).
    t, s : array_like, shape (N, K)
        Table points: lookback time [yr], ascending from 0, and SFR [M_sun/yr] (linear between them).
    """
    xa = np.asarray(log_ages, dtype=np.float64)
    ages = 10.0 ** xa
    t = np.atleast_2d(np.asarray(t, dtype=np.float64))
    s = np.atleast_2d(np.asarray(s, dtype=np.float64))
    if t.shape != s.shape or t.shape[1] < 2:
        raise ValueError("t and s must have the same shape (N, K) with K >= 2")
    if np.any(np.diff(t, axis=1) <= 0) or np.any(t[:, 0] < 0):
        raise ValueError("table times must be strictly increasing and non-negative")
    f0, f1 = _cumulative_moments(t, s, np.broadcast_to(ages, (t.shape[0], len(ages))).copy())
    dx = np.diff(xa)
    d0 = f0[:, 1:] - f0[:, :-1]
    d1 = f1[:, 1:] - f1[:, :-1]
    w = np.zeros_like(f0)
    w[:, 1:] += (d1 - xa[:-1] * d0) / dx          # rising side of each hat
    w[:, :-1] += (xa[1:] * d0 - d1) / dx          # falling side
    w[:, 0] += f0[:, 0]                           # everything younger than the youngest SSP
    return w
