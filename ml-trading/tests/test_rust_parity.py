"""Cross-parity: the Rust core must match the Python reference bit-for-bit.

Skipped automatically when the `trading_core` extension is not built
(`cd rust/trading-core && maturin develop --release`).
"""

import numpy as np
import pytest

trading_core = pytest.importorskip("trading_core")

from ml_trading.backtest.costs import CostModel  # noqa: E402
from ml_trading.backtest.simulator import run_backtest  # noqa: E402
from ml_trading.config import RiskLimits  # noqa: E402
from ml_trading.data.providers.synthetic import SyntheticProvider  # noqa: E402
from ml_trading.features.candles import candle_features  # noqa: E402
from ml_trading.features.indicators import atr  # noqa: E402
from ml_trading.risk.engine import PortfolioState, RiskEngine, TradeProposal  # noqa: E402

_REASONS = {0: "stop", 1: "target", 2: "timeout", 3: "end"}


def _limits_dict(lim: RiskLimits) -> dict:
    return {
        "risk_per_trade_pct": lim.risk_per_trade_pct,
        "min_risk_reward": lim.min_risk_reward,
        "max_daily_loss_pct": lim.max_daily_loss_pct,
        "max_position_pct": lim.max_position_pct,
        "loss_streak_cooldown": lim.loss_streak_cooldown,
        "loss_streak_size_factor": lim.loss_streak_size_factor,
        "kelly_cap_fraction": lim.kelly_cap_fraction,
    }


@pytest.fixture(scope="module")
def bars():
    return SyntheticProvider(seed=21).fetch("TEST", "1m", "2025-01-06", "2025-02-07")


def test_backtest_parity(bars) -> None:
    rng = np.random.default_rng(4)
    n = bars.height
    signals = np.zeros(n, dtype=np.int8)
    fire = rng.random(n) < 0.02
    signals[fire] = rng.choice(np.array([-1, 1], dtype=np.int8), size=int(fire.sum()))
    mults = np.clip(rng.random(n) + 0.3, 0.0, 1.0)

    lim = RiskLimits()
    costs = CostModel()
    atr_vals = bars.select(atr(14)).to_series().to_numpy()

    py_res = run_backtest(bars, signals, lim, costs, atr_vals, size_multipliers=mults)

    days = bars["ts"].dt.date().dt.epoch(time_unit="d").to_numpy().astype(np.int64)
    rs_curve, rs_trades = trading_core.run_backtest(
        bars["open"].to_numpy().astype(np.float64),
        bars["high"].to_numpy().astype(np.float64),
        bars["low"].to_numpy().astype(np.float64),
        bars["close"].to_numpy().astype(np.float64),
        days,
        signals,
        atr_vals,
        mults,
        _limits_dict(lim),
        costs.commission_per_share,
        costs.min_commission,
        costs.slippage_bps,
        100_000.0,
        1.0,
        2.0,
        60,
    )

    # bit-for-bit equity curve
    np.testing.assert_array_equal(rs_curve, py_res.equity_curve)

    assert len(rs_trades) == len(py_res.trades) > 10
    for rt, pt in zip(rs_trades, py_res.trades, strict=True):
        entry_idx, exit_idx, side, qty, entry_price, exit_price, pnl, reason = rt
        assert entry_idx == pt.entry_idx
        assert exit_idx == pt.exit_idx
        assert (side == 1) == (pt.side == "long")
        assert qty == pt.qty
        assert entry_price == pt.entry_price
        assert exit_price == pt.exit_price
        assert pnl == pt.pnl
        assert _REASONS[reason] == pt.exit_reason


def test_risk_evaluation_parity() -> None:
    lim = RiskLimits()
    engine = RiskEngine(lim)
    rng = np.random.default_rng(5)
    for _ in range(500):
        entry = 50.0 + rng.random() * 100
        atr_val = entry * (0.001 + rng.random() * 0.02)
        is_long = rng.random() < 0.5
        if is_long:
            stop, target = entry - atr_val, entry + rng.random() * 4 * atr_val
        else:
            stop, target = entry + atr_val, entry - rng.random() * 4 * atr_val
        mult = rng.random() * 1.4  # sometimes >1 to exercise clipping
        equity = 50_000 + rng.random() * 100_000
        day_start = equity / (1.0 - rng.random() * 0.05)
        losses = int(rng.integers(0, 6))

        state = PortfolioState(equity=equity, day_start_equity=day_start,
                               consecutive_losses=losses)
        py = engine.evaluate(
            TradeProposal(ticker="X", side="long" if is_long else "short",
                          entry=entry, stop=stop, target=target, size_multiplier=mult),
            state,
        )
        rs_approved, rs_qty, rs_risk, rs_rr = trading_core.evaluate_trade(
            is_long, entry, stop, target, mult, equity, day_start, losses, _limits_dict(lim)
        )
        assert rs_approved == py.approved
        assert rs_qty == py.qty
        assert rs_risk == py.risk_amount
        assert rs_rr == py.rr_ratio


def test_candle_pattern_parity(bars) -> None:
    py = candle_features(bars)
    rs = trading_core.candle_patterns(
        bars["open"].to_numpy().astype(np.float64),
        bars["high"].to_numpy().astype(np.float64),
        bars["low"].to_numpy().astype(np.float64),
        bars["close"].to_numpy().astype(np.float64),
    )
    for col in ("cdl_direction", "cdl_doji", "cdl_engulfing", "cdl_marubozu",
                "cdl_piercing_dcc", "cdl_harami", "cdl_hammer_star"):
        py_vals = py[col].fill_null(0).to_numpy().astype(np.int8)
        np.testing.assert_array_equal(rs[col], py_vals, err_msg=col)
    for col in ("cdl_body_frac", "cdl_upper_frac", "cdl_lower_frac"):
        np.testing.assert_allclose(rs[col], py[col].to_numpy(), rtol=1e-12, err_msg=col)
    gap_py = py["cdl_gap"].to_numpy()
    np.testing.assert_allclose(rs["cdl_gap"][1:], gap_py[1:], rtol=1e-12)
    assert np.isnan(rs["cdl_gap"][0]) and np.isnan(gap_py[0])
