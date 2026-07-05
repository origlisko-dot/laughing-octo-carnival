"""Deterministic risk core. Every order must pass through `RiskEngine.evaluate`.

Hard rules ML may inform but never override:
  - fixed-fraction position sizing derived from the stop distance,
  - minimum risk/reward ratio or the trade is rejected,
  - exposure caps (open positions, per-position size, sector),
  - daily-loss kill switch,
  - size reduction after a losing streak,
  - fractional-Kelly ceiling on the risk fraction.

ML/DRL layers pass a `size_multiplier` in (0, 1]; it can only shrink a trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ml_trading.config import RiskLimits


class Rejection(str, Enum):
    RISK_REWARD = "risk_reward_below_minimum"
    KILL_SWITCH = "daily_loss_kill_switch"
    MAX_POSITIONS = "max_open_positions"
    SECTOR_EXPOSURE = "sector_exposure_cap"
    BAD_LEVELS = "invalid_stop_or_target_levels"
    TOO_SMALL = "position_rounds_to_zero"


@dataclass
class TradeProposal:
    ticker: str
    side: str  # "long" or "short"
    entry: float
    stop: float
    target: float
    sector: str = "unknown"
    size_multiplier: float = 1.0  # from meta-labeling / DRL; clipped to (0,1]


@dataclass
class PositionInfo:
    ticker: str
    qty: int
    entry: float
    sector: str = "unknown"


@dataclass
class PortfolioState:
    equity: float
    day_start_equity: float
    open_positions: list[PositionInfo] = field(default_factory=list)
    consecutive_losses: int = 0


@dataclass
class RiskDecision:
    approved: bool
    qty: int = 0
    risk_amount: float = 0.0
    rr_ratio: float = 0.0
    rejection: Rejection | None = None
    notes: list[str] = field(default_factory=list)


class RiskEngine:
    def __init__(self, limits: RiskLimits) -> None:
        self.limits = limits

    def kill_switch_active(self, state: PortfolioState) -> bool:
        dd = (state.day_start_equity - state.equity) / state.day_start_equity * 100.0
        return dd >= self.limits.max_daily_loss_pct

    def evaluate(self, proposal: TradeProposal, state: PortfolioState) -> RiskDecision:
        lim = self.limits

        if proposal.side == "long":
            risk_per_share = proposal.entry - proposal.stop
            reward_per_share = proposal.target - proposal.entry
        else:
            risk_per_share = proposal.stop - proposal.entry
            reward_per_share = proposal.entry - proposal.target
        if risk_per_share <= 0 or reward_per_share <= 0:
            return RiskDecision(approved=False, rejection=Rejection.BAD_LEVELS)

        rr = reward_per_share / risk_per_share
        if rr < lim.min_risk_reward:
            return RiskDecision(approved=False, rr_ratio=rr, rejection=Rejection.RISK_REWARD)

        if self.kill_switch_active(state):
            return RiskDecision(approved=False, rr_ratio=rr, rejection=Rejection.KILL_SWITCH)

        if len(state.open_positions) >= lim.max_open_positions:
            return RiskDecision(approved=False, rr_ratio=rr, rejection=Rejection.MAX_POSITIONS)

        sector_value = sum(
            p.qty * p.entry for p in state.open_positions if p.sector == proposal.sector
        )
        if sector_value / state.equity * 100.0 >= lim.max_sector_exposure_pct:
            return RiskDecision(approved=False, rr_ratio=rr, rejection=Rejection.SECTOR_EXPOSURE)

        notes: list[str] = []
        risk_fraction = lim.risk_per_trade_pct / 100.0
        # Fractional-Kelly ceiling given the enforced R/R (conservative p=0.5 assumption).
        kelly = max((0.5 * (rr + 1.0) - 1.0) / rr, 0.0)
        if kelly > 0:
            capped = min(risk_fraction, lim.kelly_cap_fraction * kelly)
            if capped < risk_fraction:
                notes.append(f"risk fraction capped by fractional Kelly: {capped:.4f}")
            risk_fraction = capped

        if state.consecutive_losses >= lim.loss_streak_cooldown:
            risk_fraction *= lim.loss_streak_size_factor
            notes.append(f"loss streak ({state.consecutive_losses}): size x{lim.loss_streak_size_factor}")

        mult = min(max(proposal.size_multiplier, 0.0), 1.0)
        if mult < proposal.size_multiplier:
            notes.append("size_multiplier clipped to 1.0")
        risk_amount = state.equity * risk_fraction * mult

        qty = int(risk_amount / risk_per_share)
        max_qty_by_value = int(state.equity * self.limits.max_position_pct / 100.0 / proposal.entry)
        if qty > max_qty_by_value:
            qty = max_qty_by_value
            notes.append("qty capped by max position value")
        if qty <= 0:
            return RiskDecision(approved=False, rr_ratio=rr, rejection=Rejection.TOO_SMALL, notes=notes)

        return RiskDecision(
            approved=True,
            qty=qty,
            risk_amount=qty * risk_per_share,
            rr_ratio=rr,
            notes=notes,
        )


def atr_levels(
    entry: float,
    atr_value: float,
    side: str = "long",
    stop_mult: float = 1.0,
    target_mult: float = 2.0,
) -> tuple[float, float]:
    """ATR-based (stop, target) levels around an entry price."""
    if side == "long":
        return entry - stop_mult * atr_value, entry + target_mult * atr_value
    return entry + stop_mult * atr_value, entry - target_mult * atr_value
