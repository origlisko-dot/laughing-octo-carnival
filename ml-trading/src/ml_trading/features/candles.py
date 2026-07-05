"""Japanese candlestick pattern features, implemented natively on Polars.

Each pattern column is -1 (bearish), 0 (absent), or +1 (bullish). Patterns use
only the current and previous bars — no lookahead.
"""

from __future__ import annotations

import polars as pl

_BODY = (pl.col("close") - pl.col("open")).abs()
_RANGE = (pl.col("high") - pl.col("low")).clip(lower_bound=1e-12)
_UPPER = pl.col("high") - pl.max_horizontal("open", "close")
_LOWER = pl.min_horizontal("open", "close") - pl.col("low")
_BULL = pl.col("close") > pl.col("open")
_BEAR = pl.col("close") < pl.col("open")


def _sign(bull: pl.Expr, bear: pl.Expr) -> pl.Expr:
    return pl.when(bull).then(1).when(bear).then(-1).otherwise(0).cast(pl.Int8)


def candle_features(df: pl.DataFrame) -> pl.DataFrame:
    """Append candle anatomy + pattern columns to an OHLCV frame sorted by ts."""
    o, c = pl.col("open"), pl.col("close")
    body_frac = _BODY / _RANGE
    prev = {n: pl.col(n).shift(1) for n in ("open", "high", "low", "close")}
    prev_body = (prev["close"] - prev["open"]).abs()
    prev_bull = prev["close"] > prev["open"]
    prev_bear = prev["close"] < prev["open"]

    doji = body_frac < 0.1
    hammer_shape = (_LOWER > 2.0 * _BODY) & (_UPPER < _BODY)
    shooting_shape = (_UPPER > 2.0 * _BODY) & (_LOWER < _BODY)
    downtrend = pl.col("close").shift(1) < pl.col("close").shift(1).rolling_mean(10)
    uptrend = pl.col("close").shift(1) > pl.col("close").shift(1).rolling_mean(10)

    engulf_bull = _BULL & prev_bear & (c >= prev["open"]) & (o <= prev["close"]) & (_BODY > prev_body)
    engulf_bear = _BEAR & prev_bull & (c <= prev["open"]) & (o >= prev["close"]) & (_BODY > prev_body)

    marubozu = body_frac > 0.95
    prev_mid = (prev["open"] + prev["close"]) / 2
    piercing = _BULL & prev_bear & (o < prev["close"]) & (c > prev_mid) & (c < prev["open"])
    dark_cloud = _BEAR & prev_bull & (o > prev["close"]) & (c < prev_mid) & (c > prev["open"])

    harami_bull = _BULL & prev_bear & (o > prev["close"]) & (c < prev["open"]) & (_BODY < prev_body)
    harami_bear = _BEAR & prev_bull & (o < prev["close"]) & (c > prev["open"]) & (_BODY < prev_body)

    return df.with_columns(
        body_frac.alias("cdl_body_frac"),
        (_UPPER / _RANGE).alias("cdl_upper_frac"),
        (_LOWER / _RANGE).alias("cdl_lower_frac"),
        _sign(_BULL, _BEAR).alias("cdl_direction"),
        doji.cast(pl.Int8).alias("cdl_doji"),
        _sign(hammer_shape & downtrend, shooting_shape & uptrend).alias("cdl_hammer_star"),
        _sign(engulf_bull, engulf_bear).alias("cdl_engulfing"),
        _sign(marubozu & _BULL, marubozu & _BEAR).alias("cdl_marubozu"),
        _sign(piercing, dark_cloud).alias("cdl_piercing_dcc"),
        _sign(harami_bull, harami_bear).alias("cdl_harami"),
        # gap vs previous bar's range
        ((o - prev["close"]) / _RANGE).alias("cdl_gap"),
    )
