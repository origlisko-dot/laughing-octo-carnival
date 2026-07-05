"""Technical indicators as Polars expressions. Every feature uses only past data
(rolling/ewm windows end at the current bar), so the frame stays lookahead-free."""

from __future__ import annotations

import polars as pl


def _ema(col: str, span: int) -> pl.Expr:
    return pl.col(col).ewm_mean(span=span, adjust=False)


def _wilder(expr: pl.Expr, window: int) -> pl.Expr:
    return expr.ewm_mean(alpha=1.0 / window, adjust=False)


def rsi(window: int = 14) -> pl.Expr:
    delta = pl.col("close").diff()
    gain = _wilder(delta.clip(lower_bound=0.0), window)
    loss = _wilder((-delta).clip(lower_bound=0.0), window)
    return (100.0 - 100.0 / (1.0 + gain / loss)).alias(f"rsi_{window}")


def true_range() -> pl.Expr:
    prev_close = pl.col("close").shift(1)
    return pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )


def atr(window: int = 14) -> pl.Expr:
    return _wilder(true_range(), window).alias(f"atr_{window}")


def indicator_features(
    df: pl.DataFrame,
    rsi_window: int = 14,
    atr_window: int = 14,
    bb_window: int = 20,
    structure_window: int = 50,
) -> pl.DataFrame:
    """Append indicator columns to an OHLCV frame (must be sorted by ts)."""
    sma = pl.col("close").rolling_mean(bb_window)
    std = pl.col("close").rolling_std(bb_window)
    macd_line = _ema("close", 12) - _ema("close", 26)
    roll_high = pl.col("high").rolling_max(structure_window)
    roll_low = pl.col("low").rolling_min(structure_window)

    out = df.with_columns(
        rsi(rsi_window),
        atr(atr_window),
        macd_line.alias("macd"),
        macd_line.ewm_mean(span=9, adjust=False).alias("macd_signal"),
        (macd_line - macd_line.ewm_mean(span=9, adjust=False)).alias("macd_hist"),
        ((pl.col("close") - sma) / (2.0 * std)).alias("bb_pctb"),
        (4.0 * std / sma).alias("bb_width"),
        # Session-anchored VWAP: resets each trading day.
        (
            (pl.col("close") * pl.col("volume")).cum_sum().over(pl.col("ts").dt.date())
            / pl.col("volume").cum_sum().over(pl.col("ts").dt.date())
        ).alias("vwap"),
        (roll_high.alias("roll_high")),
        (roll_low.alias("roll_low")),
        (pl.col("close").pct_change().alias("ret_1")),
        (pl.col("close").pct_change(5).alias("ret_5")),
        (pl.col("close").pct_change(20).alias("ret_20")),
        (pl.col("volume") / pl.col("volume").rolling_mean(bb_window)).alias("vol_ratio"),
    )
    return out.with_columns(
        ((pl.col("close") - pl.col("vwap")) / pl.col("vwap")).alias("vwap_dist"),
        ((pl.col("roll_high") - pl.col("close")) / pl.col("close")).alias("dist_to_high"),
        ((pl.col("close") - pl.col("roll_low")) / pl.col("close")).alias("dist_to_low"),
        (pl.col("close") / pl.col("close").ewm_mean(span=structure_window, adjust=False) - 1.0).alias(
            "trend_strength"
        ),
    ).drop("roll_high", "roll_low")
