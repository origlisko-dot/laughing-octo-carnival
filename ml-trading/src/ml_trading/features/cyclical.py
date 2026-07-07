"""Cyclical / seasonality features: calendar encodings plus Fourier terms that let
tree and linear models pick up periodic behavior (time-of-day, day-of-week)."""

from __future__ import annotations

import math

import polars as pl

_DAY_MINUTES = 24 * 60
_WEEK_DAYS = 7


def cyclical_features(df: pl.DataFrame, fourier_orders: int = 2) -> pl.DataFrame:
    minute_of_day = pl.col("ts").dt.hour() * 60 + pl.col("ts").dt.minute()
    day_of_week = pl.col("ts").dt.weekday()  # 1..7

    cols: list[pl.Expr] = [
        minute_of_day.alias("t_minute_of_day"),
        day_of_week.alias("t_day_of_week"),
        pl.col("ts").dt.day().alias("t_day_of_month"),
        pl.col("ts").dt.month().alias("t_month"),
    ]
    for k in range(1, fourier_orders + 1):
        w_day = 2.0 * math.pi * k / _DAY_MINUTES
        w_week = 2.0 * math.pi * k / _WEEK_DAYS
        cols += [
            (minute_of_day * w_day).sin().alias(f"t_sin_day_{k}"),
            (minute_of_day * w_day).cos().alias(f"t_cos_day_{k}"),
            (day_of_week * w_week).sin().alias(f"t_sin_week_{k}"),
            (day_of_week * w_week).cos().alias(f"t_cos_week_{k}"),
        ]
    return df.with_columns(cols)
