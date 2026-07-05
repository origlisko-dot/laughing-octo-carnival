"""Fractional differentiation (fixed-width window FFD), after Lopez de Prado (AFML ch.5).

Ordinary differencing (d=1) makes a price series stationary but erases its long
memory. FFD with 0<d<1 keeps the series stationary *and* correlated with the
original level. `min_ffd_order` searches for the smallest d passing an ADF test.
"""

from __future__ import annotations

import numpy as np


def ffd_weights(d: float, threshold: float = 1e-5, max_width: int = 10_000) -> np.ndarray:
    """Fixed-width FFD weights, most recent first: w0=1, w_k = -w_{k-1} * (d-k+1)/k."""
    w = [1.0]
    for k in range(1, max_width):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
    return np.asarray(w)


def frac_diff_ffd(x: np.ndarray, d: float, threshold: float = 1e-5) -> np.ndarray:
    """Apply FFD to a 1-D series. First len(weights)-1 values are NaN (insufficient history)."""
    x = np.asarray(x, dtype=float)
    w = ffd_weights(d, threshold)
    width = len(w)
    out = np.full_like(x, np.nan)
    if width > len(x):
        return out
    # Convolution of the reversed weight vector over a sliding window.
    windows = np.lib.stride_tricks.sliding_window_view(x, width)
    out[width - 1 :] = windows @ w[::-1]
    return out


def adf_stat(x: np.ndarray, max_lag: int = 1) -> float:
    """Augmented Dickey-Fuller t-statistic (constant, fixed lag) — dependency-free.

    Regress dx_t on [1, x_{t-1}, dx_{t-1}, ..., dx_{t-max_lag}]; return t-stat of x_{t-1}.
    More negative => more stationary. The 5% critical value is about -2.86.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    dx = np.diff(x)
    if len(dx) <= max_lag + 3:
        return 0.0
    y = dx[max_lag:]
    cols = [np.ones_like(y), x[max_lag:-1]]
    for lag in range(1, max_lag + 1):
        cols.append(dx[max_lag - lag : -lag])
    X = np.column_stack(cols)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    sigma2 = resid @ resid / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    return float(beta[1] / np.sqrt(cov[1, 1]))


def min_ffd_order(
    x: np.ndarray,
    d_grid: np.ndarray | None = None,
    adf_threshold: float = -2.86,
) -> float:
    """Smallest d in the grid whose FFD series passes the ADF test (5% level).

    Returns 1.0 (plain differencing) if nothing smaller passes.
    """
    if d_grid is None:
        d_grid = np.arange(0.1, 1.0, 0.05)
    for d in d_grid:
        series = frac_diff_ffd(x, float(d))
        if np.isnan(series).all():
            continue
        if adf_stat(series) < adf_threshold:
            return float(d)
    return 1.0
