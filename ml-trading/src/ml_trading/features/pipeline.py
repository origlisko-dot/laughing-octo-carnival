"""Feature pipeline: raw bars -> full model-ready feature frame."""

from __future__ import annotations

import polars as pl

from ml_trading.features.candles import candle_features
from ml_trading.features.cyclical import cyclical_features
from ml_trading.features.fracdiff import frac_diff_ffd, min_ffd_order
from ml_trading.features.indicators import indicator_features
from ml_trading.features.mtf import align_multi_timeframe

NON_FEATURE_COLS = ("ts", "open", "high", "low", "close", "volume")


def base_features(df: pl.DataFrame, ffd_d: float | None = None) -> pl.DataFrame:
    """Indicators + candles + cyclical + fractionally differentiated log price for one interval."""
    out = indicator_features(df.sort("ts"))
    out = candle_features(out)
    out = cyclical_features(out)

    log_close = out.select(pl.col("close").log()).to_series().to_numpy()
    d = ffd_d if ffd_d is not None else min_ffd_order(log_close)
    out = out.with_columns(
        pl.Series("ffd_close", frac_diff_ffd(log_close, d)),
        pl.lit(d).alias("ffd_d"),
    )
    return out


def build_feature_frame(
    bars_by_interval: dict[str, pl.DataFrame],
    base_interval: str,
    context_intervals: list[str] | None = None,
    ffd_d: float | None = None,
) -> pl.DataFrame:
    """Full multi-timeframe feature frame for one ticker.

    bars_by_interval: {interval: OHLCV frame}; must include `base_interval`.
    Context frames get indicator features and are asof-joined without lookahead.
    """
    base = base_features(bars_by_interval[base_interval], ffd_d=ffd_d)
    ctx_ivs = context_intervals or [iv for iv in bars_by_interval if iv != base_interval]
    context = {
        iv: indicator_features(bars_by_interval[iv].sort("ts"))
        for iv in ctx_ivs
        if iv in bars_by_interval
    }
    ctx_cols = ["rsi_14", "macd_hist", "trend_strength", "ret_1", "bb_pctb", "vol_ratio"]
    return align_multi_timeframe(base, context, context_cols=ctx_cols)


def feature_columns(df: pl.DataFrame) -> list[str]:
    """Names of model-input columns (everything except raw OHLCV/ts)."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]
