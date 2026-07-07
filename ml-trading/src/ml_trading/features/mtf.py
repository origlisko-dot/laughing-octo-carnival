"""Multi-timeframe alignment — the heart of model 1, built to be lookahead-free.

A higher-timeframe bar labeled with open time `t` only becomes known when it
*closes* at `t + interval`. We therefore shift each context frame to its
availability time and asof-join backward, so every base-interval sample sees
the latest *completed* higher-timeframe bar and nothing newer.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from ml_trading.config import INTERVAL_SECONDS


def _availability_ts(df: pl.DataFrame, interval: str) -> pl.DataFrame:
    if interval == "1M":
        # Calendar months vary in length; a monthly bar is known at the next month's open.
        avail = pl.col("ts").dt.offset_by("1mo")
    else:
        avail = pl.col("ts") + timedelta(seconds=INTERVAL_SECONDS[interval])
    return df.with_columns(avail.alias("_avail_ts")).sort("_avail_ts")


def align_multi_timeframe(
    base: pl.DataFrame,
    context: dict[str, pl.DataFrame],
    context_cols: list[str] | None = None,
) -> pl.DataFrame:
    """Join higher-timeframe feature columns onto the base frame.

    base:     feature frame at the trading interval (e.g. 1m/5m), sorted by ts.
    context:  {interval: feature frame} for coarser intervals (e.g. 1h/1d/1w).
    context_cols: columns to carry over (default: every non-OHLCV column).

    Context columns are suffixed with the interval, e.g. `rsi_14_1h`.
    """
    out = base.sort("ts")
    for interval, ctx in context.items():
        ctx = _availability_ts(ctx, interval)
        cols = context_cols or [
            c for c in ctx.columns if c not in ("ts", "_avail_ts", "open", "high", "low", "close", "volume")
        ]
        renamed = ctx.select(
            pl.col("_avail_ts"),
            *[pl.col(c).alias(f"{c}_{interval}") for c in cols],
        )
        out = out.join_asof(renamed, left_on="ts", right_on="_avail_ts", strategy="backward").drop(
            "_avail_ts"
        )
    return out
