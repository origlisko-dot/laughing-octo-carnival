"""Benchmark: Python reference vs Rust core on the backtest hot loop.

Usage:  .venv/bin/python scripts/bench_rust.py [n_days]
"""

from __future__ import annotations

import sys
import time

import numpy as np
import trading_core

from ml_trading.backtest.costs import CostModel
from ml_trading.backtest.simulator import run_backtest
from ml_trading.config import RiskLimits
from ml_trading.data.providers.synthetic import SyntheticProvider
from ml_trading.features.indicators import atr


def main() -> None:
    n_days = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    end_month = 1 + n_days // 21
    bars = SyntheticProvider(seed=99).fetch(
        "BENCH", "1m", "2024-01-01", f"2024-{min(end_month, 12):02d}-28"
    )
    n = bars.height
    rng = np.random.default_rng(0)
    signals = np.where(rng.random(n) < 0.02, rng.choice([-1, 1], size=n), 0).astype(np.int8)
    mults = np.ones(n)
    lim = RiskLimits()
    costs = CostModel()
    atr_vals = bars.select(atr(14)).to_series().to_numpy()
    limits_dict = {
        "risk_per_trade_pct": lim.risk_per_trade_pct,
        "min_risk_reward": lim.min_risk_reward,
        "max_daily_loss_pct": lim.max_daily_loss_pct,
        "max_position_pct": lim.max_position_pct,
        "loss_streak_cooldown": lim.loss_streak_cooldown,
        "loss_streak_size_factor": lim.loss_streak_size_factor,
        "kelly_cap_fraction": lim.kelly_cap_fraction,
    }

    t0 = time.perf_counter()
    py_res = run_backtest(bars, signals, lim, costs, atr_vals, size_multipliers=mults)
    t_py = time.perf_counter() - t0

    days = bars["ts"].dt.date().dt.epoch(time_unit="d").to_numpy().astype(np.int64)
    args = (
        bars["open"].to_numpy().astype(np.float64),
        bars["high"].to_numpy().astype(np.float64),
        bars["low"].to_numpy().astype(np.float64),
        bars["close"].to_numpy().astype(np.float64),
        days, signals, atr_vals, mults, limits_dict,
        costs.commission_per_share, costs.min_commission, costs.slippage_bps,
        100_000.0, 1.0, 2.0, 60,
    )
    trading_core.run_backtest(*args)  # warm-up
    t0 = time.perf_counter()
    rs_curve, rs_trades = trading_core.run_backtest(*args)
    t_rs = time.perf_counter() - t0

    assert np.array_equal(rs_curve, py_res.equity_curve), "parity broken!"
    print(f"bars:            {n:,}")
    print(f"trades:          {len(rs_trades)}")
    print(f"python:          {t_py * 1000:9.1f} ms")
    print(f"rust:            {t_rs * 1000:9.1f} ms")
    print(f"speedup:         {t_py / t_rs:9.1f}x")


if __name__ == "__main__":
    main()
