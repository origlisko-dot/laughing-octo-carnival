import numpy as np
import polars as pl
import pytest

from ml_trading.backtest.costs import CostModel
from ml_trading.backtest.fusion import FusionConfig, fuse, fused_scores, technical_score
from ml_trading.backtest.metrics import performance_metrics
from ml_trading.backtest.report import render_report
from ml_trading.backtest.simulator import run_backtest
from ml_trading.backtest.walkforward import summarize, walk_forward_backtest
from ml_trading.config import AppConfig, RiskLimits
from ml_trading.data.providers.synthetic import SyntheticProvider
from ml_trading.features.indicators import atr


@pytest.fixture(scope="module")
def bars() -> pl.DataFrame:
    return SyntheticProvider(seed=9).fetch("TEST", "5m", "2025-01-06", "2025-02-28")


def _atr(bars: pl.DataFrame) -> np.ndarray:
    return bars.select(atr(14)).to_series().to_numpy()


def test_costs() -> None:
    cm = CostModel(commission_per_share=0.005, min_commission=1.0, slippage_bps=10.0)
    assert cm.commission(100) == 1.0  # min binds
    assert cm.commission(1000) == 5.0
    assert cm.fill_price(100.0, "buy") == pytest.approx(100.10)
    assert cm.fill_price(100.0, "sell") == pytest.approx(99.90)


def test_backtest_no_signals_flat(bars) -> None:
    res = run_backtest(
        bars, np.zeros(len(bars), dtype=np.int8), RiskLimits(), CostModel(), _atr(bars)
    )
    assert res.trades == []
    assert res.final_equity == pytest.approx(100_000.0)


def test_backtest_executes_and_costs_bite(bars) -> None:
    signals = np.zeros(len(bars), dtype=np.int8)
    signals[50:2000:100] = 1  # periodic longs
    res = run_backtest(bars, signals, RiskLimits(), CostModel(), _atr(bars))
    assert len(res.trades) > 5
    for t in res.trades:
        assert t.exit_idx >= t.entry_idx  # same-bar stop-out is legitimate
        assert t.exit_reason in ("stop", "target", "timeout", "end")
        assert t.qty > 0
    # same run with zero costs must end with strictly more equity
    free = CostModel(commission_per_share=0.0, min_commission=0.0, slippage_bps=0.0)
    res_free = run_backtest(bars, signals, RiskLimits(), free, _atr(bars))
    assert res_free.final_equity > res.final_equity


def test_backtest_entry_is_next_bar_open(bars) -> None:
    """Entry must occur at the open of the bar AFTER the signal bar."""
    signals = np.zeros(len(bars), dtype=np.int8)
    signals[100] = 1
    res = run_backtest(bars, signals, RiskLimits(), CostModel(slippage_bps=0.0), _atr(bars))
    assert res.trades
    t = res.trades[0]
    assert t.entry_idx == 101
    assert t.entry_price == pytest.approx(bars["open"][101])


def test_metrics_shapes(bars) -> None:
    signals = np.zeros(len(bars), dtype=np.int8)
    signals[50:3000:50] = 1
    res = run_backtest(bars, signals, RiskLimits(), CostModel(), _atr(bars))
    m = performance_metrics(res, bars_per_year=252 * 78)
    assert m.n_trades == len(res.trades)
    assert 0.0 <= m.win_rate <= 1.0
    assert m.max_drawdown_pct >= 0.0


def test_fusion_logic() -> None:
    cfg = FusionConfig(w_technical=0.7, w_event=0.3, entry_threshold=0.25)
    proba = np.array([[0.1, 0.2, 0.7], [0.7, 0.2, 0.1], [0.34, 0.33, 0.33]])
    tech = technical_score(proba)
    assert tech[0] > 0.5 and tech[1] < -0.5 and abs(tech[2]) < 0.05

    ev = np.array([0.5, 0.0, 0.0])
    sig = fuse(tech, ev, cfg)
    assert sig[0] == 1 and sig[1] == -1 and sig[2] == 0

    # conflict veto: strong bearish event against bullish technicals cancels
    sig_conflict = fuse(np.array([0.6]), np.array([-0.9]), cfg)
    assert sig_conflict[0] == 0

    scores = fused_scores(tech, ev, cfg)
    assert scores.shape == tech.shape and (scores >= 0).all() and (scores <= 1).all()


def test_walk_forward_and_report(bars, tmp_path) -> None:
    cfg = AppConfig()
    cfg.cv.n_splits = 3
    cfg.cv.embargo_bars = 24
    cfg.labels.max_holding_bars = 24
    outcomes = walk_forward_backtest(bars, cfg, signal_threshold=0.45,
                                     model_params={"n_estimators": 60})
    assert len(outcomes) >= 2
    # test windows must not overlap and must be ordered
    for a, b in zip(outcomes, outcomes[1:]):
        assert a.test_end <= b.test_start

    s = summarize(outcomes)
    assert "sharpe" in s and "n_trades" in s

    path = render_report(outcomes, "walk-forward TEST", tmp_path / "report.html")
    text = path.read_text()
    assert "sharpe" in text and "fold 1" in text and "<svg" in text
