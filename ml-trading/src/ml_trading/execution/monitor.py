"""Trade logging and live-vs-backtest drift monitoring."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


@dataclass
class TradeRecord:
    ts: str
    ticker: str
    side: str
    qty: int
    entry: float
    stop: float
    target: float
    signal_source: str  # "technical" | "event" | "fused"
    order_id: str
    notes: str = ""


class TradeLog:
    """Append-only JSONL trade journal — the source of truth for live performance."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, rec: TradeRecord) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(rec)) + "\n")

    def load(self) -> list[TradeRecord]:
        if not self.path.exists():
            return []
        return [TradeRecord(**json.loads(line)) for line in self.path.read_text().splitlines() if line]

    @staticmethod
    def now() -> str:
        return datetime.now(UTC).isoformat()


def drift_report(live_returns: np.ndarray, backtest_returns: np.ndarray) -> dict:
    """Compare live trade R-multiples against the backtest distribution.

    Large drift = the live market no longer matches the model's training world;
    that is the signal to halt and retrain, not to hope.
    """
    live = np.asarray(live_returns, dtype=float)
    bt = np.asarray(backtest_returns, dtype=float)
    if len(live) < 5 or len(bt) < 5:
        return {"status": "insufficient_data", "n_live": len(live), "n_backtest": len(bt)}

    mean_gap = float(live.mean() - bt.mean())
    pooled = np.sqrt((live.var() + bt.var()) / 2) or 1e-12
    effect_size = mean_gap / pooled
    # KS statistic (dependency-free)
    grid = np.sort(np.concatenate([live, bt]))
    cdf_l = np.searchsorted(np.sort(live), grid, side="right") / len(live)
    cdf_b = np.searchsorted(np.sort(bt), grid, side="right") / len(bt)
    ks = float(np.abs(cdf_l - cdf_b).max())

    status = "ok"
    if abs(effect_size) > 0.5 or ks > 0.35:
        status = "drift_warning"
    if abs(effect_size) > 1.0 or ks > 0.5:
        status = "drift_critical"
    return {
        "status": status,
        "mean_gap": mean_gap,
        "effect_size": float(effect_size),
        "ks_stat": ks,
        "n_live": len(live),
        "n_backtest": len(bt),
    }
