"""Performance metrics for backtests and walk-forward evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ml_trading.backtest.simulator import BacktestResult


@dataclass
class PerformanceMetrics:
    total_return_pct: float
    sharpe: float
    sortino: float
    max_drawdown_pct: float
    profit_factor: float
    win_rate: float
    expectancy: float
    n_trades: int

    def as_dict(self) -> dict:
        return asdict(self)


def performance_metrics(
    result: BacktestResult,
    bars_per_year: int = 252 * 390,  # 1m RTH bars
) -> PerformanceMetrics:
    eq = result.equity_curve
    rets = np.diff(eq) / eq[:-1]
    rets = rets[np.isfinite(rets)]

    total_return = (eq[-1] / eq[0] - 1.0) * 100.0
    std = rets.std()
    sharpe = float(rets.mean() / std * np.sqrt(bars_per_year)) if std > 0 else 0.0
    downside = rets[rets < 0]
    dstd = downside.std() if len(downside) else 0.0
    sortino = float(rets.mean() / dstd * np.sqrt(bars_per_year)) if dstd > 0 else 0.0

    peak = np.maximum.accumulate(eq)
    max_dd = float(((eq - peak) / peak).min() * -100.0)

    pnls = np.array([t.pnl for t in result.trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    profit_factor = float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf")
    win_rate = float(len(wins) / len(pnls)) if len(pnls) else 0.0
    expectancy = float(pnls.mean()) if len(pnls) else 0.0

    return PerformanceMetrics(
        total_return_pct=float(total_return),
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown_pct=max_dd,
        profit_factor=profit_factor if np.isfinite(profit_factor) else 0.0,
        win_rate=win_rate,
        expectancy=expectancy,
        n_trades=len(result.trades),
    )
