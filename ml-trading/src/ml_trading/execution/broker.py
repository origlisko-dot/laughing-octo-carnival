"""Broker abstraction: Alpaca paper trading + an in-memory dry-run broker.

Bracket orders only — entry, stop-loss and take-profit are submitted atomically
so a position can never exist without its exits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

from ml_trading.config import Secrets

log = logging.getLogger(__name__)

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import (
        LimitOrderRequest,
        MarketOrderRequest,
        StopLossRequest,
        TakeProfitRequest,
    )
except ImportError:  # optional extra
    TradingClient = None
    OrderSide = None
    TimeInForce = None
    LimitOrderRequest = None
    MarketOrderRequest = None
    StopLossRequest = None
    TakeProfitRequest = None


@dataclass
class BracketOrder:
    ticker: str
    side: str  # "long" | "short"
    qty: int
    stop: float
    target: float
    limit: float | None = None  # None = market entry


class Broker(Protocol):
    def submit_bracket(self, order: BracketOrder) -> str: ...

    def equity(self) -> float: ...

    def open_position_count(self) -> int: ...


@dataclass
class DryRunBroker:
    """Logs orders instead of sending them; used by `mlt paper-trade --dry-run` and tests."""

    initial_equity: float = 100_000.0
    submitted: list[BracketOrder] = field(default_factory=list)

    def submit_bracket(self, order: BracketOrder) -> str:
        self.submitted.append(order)
        log.info("[dry-run] bracket %s", order)
        return f"dry-{len(self.submitted)}"

    def equity(self) -> float:
        return self.initial_equity

    def open_position_count(self) -> int:
        return 0


class AlpacaPaperBroker:
    def __init__(self, secrets: Secrets | None = None) -> None:
        if TradingClient is None:
            raise RuntimeError("alpaca-py not installed; run: uv pip install 'ml-trading[brokers]'")
        s = secrets or Secrets()
        if not s.alpaca_api_key:
            raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set in the environment")
        # paper=True is hard-coded: switching to live trading is an explicit code change,
        # never a config flip.
        self._client = TradingClient(s.alpaca_api_key, s.alpaca_secret_key, paper=True)

    def submit_bracket(self, order: BracketOrder) -> str:
        side = OrderSide.BUY if order.side == "long" else OrderSide.SELL
        common = dict(
            symbol=order.ticker,
            qty=order.qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            order_class="bracket",
            take_profit=TakeProfitRequest(limit_price=round(order.target, 2)),
            stop_loss=StopLossRequest(stop_price=round(order.stop, 2)),
        )
        if order.limit is not None:
            req = LimitOrderRequest(limit_price=round(order.limit, 2), **common)
        else:
            req = MarketOrderRequest(**common)
        result = self._client.submit_order(req)
        return str(result.id)

    def equity(self) -> float:
        return float(self._client.get_account().equity)

    def open_position_count(self) -> int:
        return len(self._client.get_all_positions())
