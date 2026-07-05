"""Event impact model: predict abnormal volatility/return after a catalyst.

Trained on historical (catalyst, market-reaction) pairs; at inference it scores
fresh catalysts to find events likely to move the stock before they are fully
priced in. Output feeds the fusion layer as the event-side signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import lightgbm as lgb
import numpy as np
import polars as pl

from ml_trading.events.classify import Catalyst, CatalystType

_TYPES = list(CatalystType)


def catalyst_features(cat: Catalyst) -> np.ndarray:
    """Encode a catalyst: one-hot type + direction + strength + source + timing."""
    onehot = np.zeros(len(_TYPES))
    onehot[_TYPES.index(cat.type)] = 1.0
    hour = cat.item.ts.hour + cat.item.ts.minute / 60.0
    return np.concatenate(
        [
            onehot,
            [
                float(cat.direction),
                cat.strength,
                1.0 if cat.item.source == "edgar" else 0.0,
                hour / 24.0,
                float(cat.item.ts.weekday()) / 6.0,
            ],
        ]
    )


@dataclass
class EventReaction:
    """Measured market reaction in a window after the event."""

    abnormal_return: float  # return minus expected drift over the window
    abnormal_vol_ratio: float  # realized vol in window / trailing baseline vol


def measure_reaction(
    bars: pl.DataFrame,
    event_ts: datetime,
    window_bars: int = 12,
    baseline_bars: int = 120,
) -> EventReaction | None:
    """Compute the post-event reaction from an OHLCV frame (event interval bars)."""
    df = bars.sort("ts")
    after = df.filter(pl.col("ts") >= event_ts).head(window_bars)
    before = df.filter(pl.col("ts") < event_ts).tail(baseline_bars)
    if after.height < 2 or before.height < 20:
        return None
    anchor = float(before["close"][-1])  # last pre-event close: captures the event jump itself
    rets_after = after.select(pl.col("close").pct_change().alias("r")).drop_nulls()["r"]
    rets_before = before.select(pl.col("close").pct_change().alias("r")).drop_nulls()["r"]
    base_vol = float(rets_before.std() or 1e-9)
    window_ret = float(after["close"][-1] / anchor - 1.0)
    drift = float(rets_before.mean() or 0.0) * len(rets_after)
    return EventReaction(
        abnormal_return=window_ret - drift,
        abnormal_vol_ratio=float((rets_after.std() or 0.0) / base_vol),
    )


class ImpactModel:
    """Two LightGBM regressors: expected abnormal return and abnormal vol ratio."""

    def __init__(self, **params) -> None:
        base = {"objective": "regression", "verbosity": -1, "n_estimators": 200}
        self._ret = lgb.LGBMRegressor(**{**base, **params})
        self._vol = lgb.LGBMRegressor(**{**base, **params})

    def fit(self, catalysts: list[Catalyst], reactions: list[EventReaction]) -> ImpactModel:
        X = np.vstack([catalyst_features(c) for c in catalysts])
        self._ret.fit(X, np.array([r.abnormal_return for r in reactions]))
        self._vol.fit(X, np.array([r.abnormal_vol_ratio for r in reactions]))
        return self

    def predict(self, cat: Catalyst) -> tuple[float, float]:
        """(expected abnormal return, expected abnormal vol ratio)."""
        x = catalyst_features(cat).reshape(1, -1)
        return float(self._ret.predict(x)[0]), float(self._vol.predict(x)[0])

    def event_signal(self, cat: Catalyst, min_vol_ratio: float = 1.5) -> float:
        """Signed signal in [-1, 1]: direction * confidence, zeroed if no abnormal vol expected."""
        exp_ret, exp_vol = self.predict(cat)
        if exp_vol < min_vol_ratio:
            return 0.0
        direction = np.sign(exp_ret) if cat.direction == 0 else cat.direction
        magnitude = min(abs(exp_ret) * 50.0, 1.0) * cat.strength
        return float(direction * magnitude)


def is_fresh(cat: Catalyst, now: datetime, max_age_minutes: int = 120) -> bool:
    """Stale events are assumed priced in; only fresh catalysts may trade."""
    return (now - cat.item.ts) <= timedelta(minutes=max_age_minutes)
