"""Cross-asset lead-lag discovery and features.

Large, liquid names often move first; smaller names in the same sector follow
minutes later. We estimate lagged cross-correlations and a Granger-style F test
per ordered pair, keep stable pairs, and expose the leader's recent abnormal
move as a real-time feature for the lagger.

Relationships drift, so `lead_lag_matrix` is meant to be re-estimated inside
every walk-forward training window — never on the full history.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class LeadLagPair:
    leader: str
    lagger: str
    lag: int  # bars by which the leader precedes the lagger
    corr: float
    f_stat: float


def _aligned_returns(bars: dict[str, pl.DataFrame]) -> pl.DataFrame:
    frames = []
    for ticker, df in bars.items():
        frames.append(df.select("ts", pl.col("close").pct_change().alias(ticker)))
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, on="ts", how="inner")
    return out.drop_nulls().sort("ts")


def granger_f_stat(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    """F statistic: do lagged x values improve prediction of y beyond y's own lags?"""
    n = len(y)
    if n <= 2 * lag + 5:
        return 0.0
    Y = y[lag:]
    own = np.column_stack([y[lag - k : n - k] for k in range(1, lag + 1)])
    cross = np.column_stack([x[lag - k : n - k] for k in range(1, lag + 1)])
    ones = np.ones((len(Y), 1))

    X_r = np.hstack([ones, own])
    X_f = np.hstack([ones, own, cross])
    rss_r = _rss(X_r, Y)
    rss_f = _rss(X_f, Y)
    df1 = lag
    df2 = len(Y) - X_f.shape[1]
    if df2 <= 0 or rss_f <= 0:
        return 0.0
    return float(max((rss_r - rss_f) / df1 / (rss_f / df2), 0.0))


def _rss(X: np.ndarray, y: np.ndarray) -> float:
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    return float(r @ r)


def lead_lag_matrix(
    bars: dict[str, pl.DataFrame],
    max_lag: int = 5,
    min_corr: float = 0.05,
    min_f: float = 4.0,
) -> list[LeadLagPair]:
    """Discover ordered lead->lag pairs from synchronized bar frames (one interval)."""
    rets = _aligned_returns(bars)
    tickers = [c for c in rets.columns if c != "ts"]
    arr = {t: rets[t].to_numpy() for t in tickers}
    pairs: list[LeadLagPair] = []
    for leader in tickers:
        for lagger in tickers:
            if leader == lagger:
                continue
            best_lag, best_corr = 0, 0.0
            x, y = arr[leader], arr[lagger]
            for lag in range(1, max_lag + 1):
                c = float(np.corrcoef(x[:-lag], y[lag:])[0, 1])
                if abs(c) > abs(best_corr):
                    best_corr, best_lag = c, lag
            if abs(best_corr) < min_corr or best_lag == 0:
                continue
            f = granger_f_stat(x, y, best_lag)
            if f >= min_f:
                pairs.append(
                    LeadLagPair(leader=leader, lagger=lagger, lag=best_lag, corr=best_corr, f_stat=f)
                )
    return sorted(pairs, key=lambda p: -p.f_stat)


def leader_features(
    lagger_df: pl.DataFrame,
    leader_df: pl.DataFrame,
    pair: LeadLagPair,
    z_window: int = 60,
) -> pl.DataFrame:
    """Append the leader's z-scored return (as of the lagger's bar time) to the lagger frame."""
    lead_ret = leader_df.select(
        "ts",
        pl.col("close").pct_change().alias("_lead_ret"),
    ).with_columns(
        (
            (pl.col("_lead_ret") - pl.col("_lead_ret").rolling_mean(z_window))
            / pl.col("_lead_ret").rolling_std(z_window)
        ).alias(f"lead_{pair.leader}_z")
    )
    return lagger_df.sort("ts").join_asof(
        lead_ret.select("ts", f"lead_{pair.leader}_z").sort("ts"),
        on="ts",
        strategy="backward",
    )
