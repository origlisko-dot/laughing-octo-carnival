"""Paper-trading loop: bars in -> features -> model -> fusion -> risk -> bracket order.

One iteration = one completed base-interval bar. Everything upstream of the
broker is identical to the backtest path, which is what makes live-vs-backtest
drift measurable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import polars as pl

from ml_trading.backtest.fusion import FusionConfig, fuse, technical_score
from ml_trading.config import AppConfig
from ml_trading.execution.broker import BracketOrder, Broker
from ml_trading.execution.imbalance import ImbalanceGuard, tick_rule_imbalance
from ml_trading.execution.monitor import TradeLog, TradeRecord
from ml_trading.features.pipeline import base_features, feature_columns
from ml_trading.models.tabular import TabularModel
from ml_trading.risk.engine import PortfolioState, RiskEngine, TradeProposal, atr_levels

log = logging.getLogger(__name__)


@dataclass
class TickDecision:
    ticker: str
    action: str  # "order" | "hold" | "veto_imbalance" | "rejected_risk"
    order_id: str = ""
    detail: str = ""


class PaperTrader:
    def __init__(
        self,
        cfg: AppConfig,
        model: TabularModel,
        broker: Broker,
        trade_log: TradeLog,
        fusion_cfg: FusionConfig | None = None,
        guard: ImbalanceGuard | None = None,
        signal_threshold: float = 0.6,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.broker = broker
        self.trade_log = trade_log
        self.fusion_cfg = fusion_cfg or FusionConfig()
        self.guard = guard or ImbalanceGuard()
        self.signal_threshold = signal_threshold
        self.engine = RiskEngine(cfg.risk)
        self._day_start_equity: float | None = None

    def on_bar(
        self,
        ticker: str,
        bars: pl.DataFrame,
        event_signal: float = 0.0,
        sector: str = "unknown",
    ) -> TickDecision:
        """Process the latest completed bar for one ticker. `bars` = recent history
        ending at the completed bar (enough rows for feature warm-up)."""
        equity = self.broker.equity()
        if self._day_start_equity is None:
            self._day_start_equity = equity
        state = PortfolioState(equity=equity, day_start_equity=self._day_start_equity)

        feats = base_features(bars)
        cols = [c for c in feature_columns(feats) if feats[c].dtype.is_numeric()]
        x = feats.select(cols).fill_nan(None).to_numpy()[-1:]
        if np.isnan(x).all():
            return TickDecision(ticker, "hold", detail="features not warmed up")

        proba = self.model.predict_proba(x)
        tech = technical_score(proba)
        signal = fuse(tech, np.array([event_signal]), self.fusion_cfg)[0]
        if signal == 0:
            return TickDecision(ticker, "hold", detail=f"tech={tech[0]:+.2f} ev={event_signal:+.2f}")

        side = "long" if signal > 0 else "short"
        imb = tick_rule_imbalance(bars["close"].to_numpy()[-30:], bars["volume"].to_numpy()[-30:])
        guard_mult = self.guard.size_multiplier(side, imb)
        if guard_mult == 0.0:
            return TickDecision(ticker, "veto_imbalance", detail=f"imbalance={imb:+.2f}")

        entry = float(bars["close"][-1])
        atr_val = float(feats[f"atr_{self.cfg.labels.atr_window}"][-1])
        stop, target = atr_levels(
            entry, atr_val, side, self.cfg.labels.stop_atr_mult, self.cfg.labels.profit_atr_mult
        )
        proposal = TradeProposal(
            ticker=ticker, side=side, entry=entry, stop=stop, target=target,
            sector=sector, size_multiplier=guard_mult,
        )
        decision = self.engine.evaluate(proposal, state)
        if not decision.approved:
            return TickDecision(ticker, "rejected_risk", detail=str(decision.rejection))

        order_id = self.broker.submit_bracket(
            BracketOrder(ticker=ticker, side=side, qty=decision.qty, stop=stop, target=target)
        )
        self.trade_log.record(
            TradeRecord(
                ts=TradeLog.now(), ticker=ticker, side=side, qty=decision.qty,
                entry=entry, stop=stop, target=target,
                signal_source="fused", order_id=order_id,
                notes="; ".join(decision.notes),
            )
        )
        log.info("order %s: %s %s x%d @ %.2f (stop %.2f, target %.2f)",
                 order_id, side, ticker, decision.qty, entry, stop, target)
        return TickDecision(ticker, "order", order_id=order_id)

    def on_new_session(self) -> None:
        self._day_start_equity = None
