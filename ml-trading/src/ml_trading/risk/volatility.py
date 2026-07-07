"""Volatility forecasting for dynamic position sizing.

EWMA is the always-available baseline; GARCH(1,1) (via the `arch` extra) and a
LightGBM forecaster on realized-vol features refine it. The risk engine consumes
a `vol_scalar`: target_vol / forecast_vol, clipped to [0.25, 1.0] so it can only
shrink size (never lever up beyond the deterministic core).
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np

try:
    from arch import arch_model
except ImportError:  # optional extra
    arch_model = None


def ewma_volatility(returns: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """RiskMetrics EWMA sigma_t (one-step-ahead, uses info through t-1)."""
    r = np.asarray(returns, dtype=float)
    var = np.empty_like(r)
    var[0] = np.var(r[: min(20, len(r))]) or 1e-8
    for t in range(1, len(r)):
        var[t] = lam * var[t - 1] + (1 - lam) * r[t - 1] ** 2
    return np.sqrt(var)


def garch_forecast(returns: np.ndarray, horizon: int = 1) -> float:
    """One-shot GARCH(1,1) sigma forecast (rescaled internally for stability)."""
    if arch_model is None:
        raise RuntimeError("arch not installed; run: uv pip install 'ml-trading[vol]'")
    scale = 100.0
    am = arch_model(np.asarray(returns) * scale, vol="GARCH", p=1, q=1, rescale=False)
    res = am.fit(disp="off")
    f = res.forecast(horizon=horizon, reindex=False)
    return float(np.sqrt(f.variance.values[-1, -1]) / scale)


class LightGBMVolForecaster:
    """Predict next-bar realized vol from lagged vol/return features."""

    def __init__(self, lags: int = 10, **params) -> None:
        self.lags = lags
        self.params = {"objective": "regression", "verbosity": -1, "n_estimators": 200, **params}
        self._model: lgb.LGBMRegressor | None = None

    def _features(self, returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = np.abs(np.asarray(returns, dtype=float))
        X, y = [], []
        for t in range(self.lags, len(r) - 1):
            X.append(np.concatenate([r[t - self.lags : t], [r[t - 5 : t].mean(), r[t - 5 : t].std()]]))
            y.append(r[t + 1])
        return np.asarray(X), np.asarray(y)

    def fit(self, returns: np.ndarray) -> LightGBMVolForecaster:
        X, y = self._features(returns)
        self._model = lgb.LGBMRegressor(**self.params).fit(X, y)
        return self

    def predict_next(self, returns: np.ndarray) -> float:
        r = np.abs(np.asarray(returns, dtype=float))
        x = np.concatenate([r[-self.lags :], [r[-5:].mean(), r[-5:].std()]])
        return float(max(self._model.predict(x.reshape(1, -1))[0], 1e-8))


def vol_scalar(forecast_vol: float, target_vol: float, lo: float = 0.25, hi: float = 1.0) -> float:
    """Size multiplier from a volatility forecast; can only de-risk (hi=1.0)."""
    if forecast_vol <= 0:
        return hi
    return float(np.clip(target_vol / forecast_vol, lo, hi))
