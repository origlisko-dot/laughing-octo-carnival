"""Trading costs: per-share commission with a minimum, plus slippage in basis points.

Every backtest must run through this model — zero-cost backtests are forbidden.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    commission_per_share: float = 0.005
    min_commission: float = 1.0
    slippage_bps: float = 2.0

    def commission(self, qty: int) -> float:
        return max(abs(qty) * self.commission_per_share, self.min_commission)

    def fill_price(self, price: float, side: str) -> float:
        """Adverse slippage: buys fill higher, sells fill lower."""
        slip = price * self.slippage_bps / 10_000.0
        return price + slip if side == "buy" else price - slip
