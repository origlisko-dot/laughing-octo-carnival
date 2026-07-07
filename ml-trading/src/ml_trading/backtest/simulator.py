"""Event-driven backtest simulator (reference implementation).

Bar-by-bar semantics, chosen to be conservative and exactly portable to the
Rust core (`trading_core.run_backtest` must match this bit-for-bit):

- A signal computed at bar i's close enters at bar i+1's *open* (no lookahead),
  with adverse slippage and commission.
- Stops/targets are evaluated intrabar on high/low; if both are touched within
  one bar, the stop wins (conservative).
- Exits also pay slippage and commission.
- Position sizing and all limits go through the deterministic RiskEngine.
- The daily-loss kill switch flattens nothing retroactively but blocks new
  entries for the rest of the session.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from ml_trading.backtest.costs import CostModel
from ml_trading.config import RiskLimits
from ml_trading.risk.engine import PortfolioState, RiskEngine, TradeProposal, atr_levels


@dataclass
class Trade:
    entry_idx: int
    exit_idx: int
    side: str
    qty: int
    entry_price: float
    exit_price: float
    pnl: float
    exit_reason: str  # "stop" | "target" | "timeout" | "end"


@dataclass
class BacktestResult:
    equity_curve: np.ndarray
    trades: list[Trade] = field(default_factory=list)

    @property
    def final_equity(self) -> float:
        return float(self.equity_curve[-1])


def run_backtest(
    bars: pl.DataFrame,
    signals: np.ndarray,
    limits: RiskLimits,
    costs: CostModel,
    atr_values: np.ndarray,
    initial_equity: float = 100_000.0,
    stop_atr_mult: float = 1.0,
    target_atr_mult: float = 2.0,
    max_holding_bars: int = 60,
    size_multipliers: np.ndarray | None = None,
) -> BacktestResult:
    """signals[i] in {-1, 0, +1}: decision at bar i close; size_multipliers optional (0..1]."""
    ts = bars["ts"].to_numpy()
    open_ = bars["open"].to_numpy().astype(np.float64)
    high = bars["high"].to_numpy().astype(np.float64)
    low = bars["low"].to_numpy().astype(np.float64)
    close = bars["close"].to_numpy().astype(np.float64)
    n = len(close)
    days = bars["ts"].dt.date().to_numpy()
    if size_multipliers is None:
        size_multipliers = np.ones(n)

    engine = RiskEngine(limits)
    state = PortfolioState(equity=initial_equity, day_start_equity=initial_equity)
    equity_curve = np.empty(n)
    trades: list[Trade] = []

    pos_qty = 0
    pos_side = ""
    pos_entry = 0.0
    pos_stop = 0.0
    pos_target = 0.0
    pos_entry_idx = -1
    pending_signal = 0
    pending_mult = 1.0

    for i in range(n):
        if i > 0 and days[i] != days[i - 1]:
            state.day_start_equity = state.equity

        # 1) execute pending entry at this bar's open
        if pending_signal != 0 and pos_qty == 0:
            side = "long" if pending_signal > 0 else "short"
            raw_entry = open_[i]
            a = atr_values[i - 1] if i > 0 else np.nan
            if np.isfinite(a) and a > 0:
                stop, target = atr_levels(raw_entry, a, side, stop_atr_mult, target_atr_mult)
                proposal = TradeProposal(
                    ticker="X", side=side, entry=raw_entry, stop=stop, target=target,
                    size_multiplier=pending_mult,
                )
                decision = engine.evaluate(proposal, state)
                if decision.approved:
                    fill = costs.fill_price(raw_entry, "buy" if side == "long" else "sell")
                    state.equity -= costs.commission(decision.qty)
                    pos_qty = decision.qty
                    pos_side = side
                    pos_entry = fill
                    pos_stop = stop
                    pos_target = target
                    pos_entry_idx = i
            pending_signal = 0

        # 2) manage open position on this bar
        if pos_qty > 0:
            exit_reason = ""
            exit_price = 0.0
            if pos_side == "long":
                if low[i] <= pos_stop:
                    exit_reason, exit_price = "stop", pos_stop
                elif high[i] >= pos_target:
                    exit_reason, exit_price = "target", pos_target
            else:
                if high[i] >= pos_stop:
                    exit_reason, exit_price = "stop", pos_stop
                elif low[i] <= pos_target:
                    exit_reason, exit_price = "target", pos_target
            if not exit_reason and i - pos_entry_idx >= max_holding_bars:
                exit_reason, exit_price = "timeout", close[i]
            if not exit_reason and i == n - 1:
                exit_reason, exit_price = "end", close[i]

            if exit_reason:
                fill = costs.fill_price(exit_price, "sell" if pos_side == "long" else "buy")
                sign = 1.0 if pos_side == "long" else -1.0
                pnl = sign * (fill - pos_entry) * pos_qty - costs.commission(pos_qty)
                state.equity += pnl
                state.consecutive_losses = 0 if pnl > 0 else state.consecutive_losses + 1
                trades.append(
                    Trade(
                        entry_idx=pos_entry_idx, exit_idx=i, side=pos_side, qty=pos_qty,
                        entry_price=pos_entry, exit_price=fill, pnl=pnl, exit_reason=exit_reason,
                    )
                )
                pos_qty = 0

        # 3) queue a new signal decided at this bar's close
        if pos_qty == 0 and signals[i] != 0 and not engine.kill_switch_active(state):
            pending_signal = int(signals[i])
            pending_mult = float(size_multipliers[i])

        # 4) mark to market
        unrealized = 0.0
        if pos_qty > 0:
            sign = 1.0 if pos_side == "long" else -1.0
            unrealized = sign * (close[i] - pos_entry) * pos_qty
        equity_curve[i] = state.equity + unrealized

    _ = ts  # timestamps kept for future multi-asset alignment
    return BacktestResult(equity_curve=equity_curve, trades=trades)
