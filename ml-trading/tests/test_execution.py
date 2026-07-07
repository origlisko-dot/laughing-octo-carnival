import numpy as np
import pytest

from ml_trading.backtest.fusion import FusionConfig
from ml_trading.config import AppConfig
from ml_trading.data.providers.synthetic import SyntheticProvider
from ml_trading.execution.broker import BracketOrder, DryRunBroker
from ml_trading.execution.imbalance import ImbalanceGuard, tick_rule_imbalance
from ml_trading.execution.monitor import TradeLog, TradeRecord, drift_report
from ml_trading.execution.runner import PaperTrader
from ml_trading.features.pipeline import base_features, feature_columns
from ml_trading.labeling.triple_barrier import triple_barrier_labels
from ml_trading.models.tabular import TabularModel


def test_tick_rule_imbalance_directions() -> None:
    up = np.array([100.0, 100.1, 100.2, 100.3])
    down = up[::-1].copy()
    vol = np.ones(4) * 100
    assert tick_rule_imbalance(up, vol) == pytest.approx(1.0)
    assert tick_rule_imbalance(down, vol) == pytest.approx(-1.0)
    flat = np.array([100.0, 100.0, 100.0])
    assert tick_rule_imbalance(flat, np.ones(3)) == 0.0


def test_imbalance_guard() -> None:
    g = ImbalanceGuard(veto_threshold=0.6, shrink_threshold=0.3)
    assert g.size_multiplier("long", -0.7) == 0.0  # heavy selling against a long: veto
    assert g.size_multiplier("long", -0.4) == 0.5  # moderate: shrink
    assert g.size_multiplier("long", 0.5) == 1.0  # flow with us: full size
    assert g.size_multiplier("short", 0.7) == 0.0  # heavy buying against a short: veto


def test_trade_log_roundtrip(tmp_path) -> None:
    log = TradeLog(tmp_path / "trades.jsonl")
    rec = TradeRecord(
        ts=TradeLog.now(), ticker="AAPL", side="long", qty=10, entry=100.0,
        stop=99.0, target=102.0, signal_source="fused", order_id="x1",
    )
    log.record(rec)
    log.record(rec)
    loaded = log.load()
    assert len(loaded) == 2
    assert loaded[0].ticker == "AAPL"


def test_drift_report_levels() -> None:
    rng = np.random.default_rng(0)
    bt = rng.normal(0.2, 1.0, 500)
    same = rng.normal(0.2, 1.0, 100)
    shifted = rng.normal(-1.5, 1.0, 100)
    assert drift_report(same, bt)["status"] == "ok"
    assert drift_report(shifted, bt)["status"] == "drift_critical"
    assert drift_report(np.ones(2), bt)["status"] == "insufficient_data"


@pytest.fixture(scope="module")
def trained_setup():
    bars = SyntheticProvider(seed=13).fetch("TEST", "5m", "2025-01-06", "2025-02-21")
    feats = base_features(bars)
    labels = triple_barrier_labels(bars, max_holding=24)
    cols = [c for c in feature_columns(feats) if feats[c].dtype.is_numeric()]
    X = feats.select(cols).fill_nan(None).to_numpy()
    model = TabularModel(params={"n_estimators": 50}).fit(X, labels.label)
    return bars, model


def test_paper_trader_flow(tmp_path, trained_setup) -> None:
    bars, model = trained_setup
    cfg = AppConfig()
    broker = DryRunBroker()
    trader = PaperTrader(
        cfg, model, broker, TradeLog(tmp_path / "t.jsonl"),
        fusion_cfg=FusionConfig(entry_threshold=0.05),  # permissive so we exercise the order path
        signal_threshold=0.5,
    )
    actions = set()
    for end in range(400, bars.height, 37):
        d = trader.on_bar("TEST", bars.head(end))
        actions.add(d.action)
    assert "order" in actions or "rejected_risk" in actions or "veto_imbalance" in actions
    if broker.submitted:
        o: BracketOrder = broker.submitted[0]
        assert o.qty > 0
        if o.side == "long":
            assert o.stop < o.target
        else:
            assert o.stop > o.target
        # every submitted order is journaled
        assert len(TradeLog(tmp_path / "t.jsonl").load()) == len(broker.submitted)


def test_paper_trader_event_veto(tmp_path, trained_setup) -> None:
    """A strong opposing event signal must veto entries (conflict veto in fusion)."""
    bars, model = trained_setup
    cfg = AppConfig()
    trader = PaperTrader(
        cfg, model, DryRunBroker(), TradeLog(tmp_path / "t2.jsonl"),
        fusion_cfg=FusionConfig(entry_threshold=0.05, conflict_veto=True),
    )
    broker = trader.broker
    for end in (500, 700, 900, 1100, 1300):
        d = trader.on_bar("TEST", bars.head(end), event_signal=0.0)
        if d.action == "order":
            side = broker.submitted[-1].side
            opposing = -0.9 if side == "long" else 0.9
            trader2 = PaperTrader(
                cfg, model, DryRunBroker(), TradeLog(tmp_path / "t3.jsonl"),
                fusion_cfg=FusionConfig(entry_threshold=0.05, conflict_veto=True),
            )
            d2 = trader2.on_bar("TEST", bars.head(end), event_signal=opposing)
            assert d2.action != "order", "opposing strong event signal must veto the entry"
            break
    else:
        pytest.skip("no order fired on sampled bars; veto path not exercisable here")
