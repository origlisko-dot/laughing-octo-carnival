"""Triple-barrier labeling (Lopez de Prado).

For each bar we look forward until one of three barriers is hit:
  +1  profit target  (entry + profit_mult * ATR)
  -1  stop           (entry - stop_mult * ATR)
   0  time expiry    (max_holding bars, sign decided by the sign of the return, but
                      kept as class 0 = "no clean move")

Also returns `label_end_idx` — the bar index where the label is resolved — which
purged CV needs to know how far information from each sample leaks forward.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ml_trading.features.indicators import atr


@dataclass
class TripleBarrierResult:
    label: np.ndarray  # int8 in {-1, 0, +1}
    label_end_idx: np.ndarray  # int64: index where each label resolves
    realized_return: np.ndarray  # signed return at barrier touch


def triple_barrier_labels(
    df: pl.DataFrame,
    profit_mult: float = 2.0,
    stop_mult: float = 1.0,
    max_holding: int = 60,
    atr_window: int = 14,
) -> TripleBarrierResult:
    """Label long-side outcomes on an OHLCV frame sorted by ts.

    Barrier touches are evaluated on future bar highs/lows (intrabar), entry at
    the close of the labeled bar. If both barriers are hit within the same
    future bar, the stop wins (conservative).
    """
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    atr_arr = df.select(atr(atr_window)).to_series().to_numpy()
    n = len(close)

    label = np.zeros(n, dtype=np.int8)
    end_idx = np.arange(n, dtype=np.int64)
    realized = np.zeros(n, dtype=np.float64)

    for i in range(n):
        a = atr_arr[i]
        if not np.isfinite(a) or a <= 0 or i + 1 >= n:
            end_idx[i] = min(i + max_holding, n - 1)
            continue
        entry = close[i]
        upper = entry + profit_mult * a
        lower = entry - stop_mult * a
        last = min(i + max_holding, n - 1)
        hit = 0
        j = last
        for k in range(i + 1, last + 1):
            if low[k] <= lower:  # stop checked first: conservative on same-bar touches
                hit, j = -1, k
                break
            if high[k] >= upper:
                hit, j = 1, k
                break
        label[i] = hit
        end_idx[i] = j
        if hit == 1:
            realized[i] = (upper - entry) / entry
        elif hit == -1:
            realized[i] = (lower - entry) / entry
        else:
            realized[i] = (close[j] - entry) / entry
    return TripleBarrierResult(label=label, label_end_idx=end_idx, realized_return=realized)
