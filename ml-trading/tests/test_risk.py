import numpy as np
import pytest

from ml_trading.config import RiskLimits
from ml_trading.risk.drl.env import SizingEnv, fixed_fraction_baseline
from ml_trading.risk.engine import (
    PortfolioState,
    PositionInfo,
    Rejection,
    RiskEngine,
    TradeProposal,
    atr_levels,
)
from ml_trading.risk.metalabel import MetaLabeler
from ml_trading.risk.volatility import LightGBMVolForecaster, ewma_volatility, vol_scalar


@pytest.fixture
def engine() -> RiskEngine:
    # max_position_pct=100 so sizing tests exercise the risk budget, not the value cap
    return RiskEngine(
        RiskLimits(
            risk_per_trade_pct=1.0, min_risk_reward=2.0, max_daily_loss_pct=3.0, max_position_pct=100.0
        )
    )


@pytest.fixture
def state() -> PortfolioState:
    return PortfolioState(equity=100_000.0, day_start_equity=100_000.0)


def proposal(**kw) -> TradeProposal:
    base = dict(ticker="AAPL", side="long", entry=100.0, stop=99.0, target=102.0, sector="tech")
    return TradeProposal(**{**base, **kw})


def test_position_sizing_matches_risk_budget(engine, state) -> None:
    d = engine.evaluate(proposal(), state)
    assert d.approved
    # 1% of 100k = $1000 risk budget at $1/share risk.
    # Kelly cap: p=0.5, rr=2 => f*=0.25, capped 0.5*0.25=0.125 > 0.01, so it doesn't bind.
    assert d.qty == 1000
    assert d.risk_amount == pytest.approx(1000.0)
    assert d.rr_ratio == pytest.approx(2.0)


def test_rr_enforcement_rejects(engine, state) -> None:
    d = engine.evaluate(proposal(target=101.5), state)  # rr = 1.5 < 2.0
    assert not d.approved
    assert d.rejection is Rejection.RISK_REWARD


def test_kill_switch(engine, state) -> None:
    state.equity = 96_500.0  # -3.5% on the day
    d = engine.evaluate(proposal(), state)
    assert not d.approved and d.rejection is Rejection.KILL_SWITCH


def test_max_positions_and_sector_caps(engine, state) -> None:
    state.open_positions = [PositionInfo("T", 100, 100.0, "tech") for _ in range(5)]
    d = engine.evaluate(proposal(), state)
    assert d.rejection is Rejection.MAX_POSITIONS

    state.open_positions = [PositionInfo("T", 350, 100.0, "tech")]  # 35% of equity in tech
    d = engine.evaluate(proposal(), state)
    assert d.rejection is Rejection.SECTOR_EXPOSURE


def test_loss_streak_shrinks_size(engine, state) -> None:
    d_full = engine.evaluate(proposal(), state)
    state.consecutive_losses = 3
    d_cut = engine.evaluate(proposal(), state)
    assert d_cut.approved
    assert d_cut.qty == d_full.qty // 2


def test_size_multiplier_only_shrinks(engine, state) -> None:
    d_half = engine.evaluate(proposal(size_multiplier=0.5), state)
    d_over = engine.evaluate(proposal(size_multiplier=5.0), state)  # must clip to 1.0
    d_full = engine.evaluate(proposal(), state)
    assert d_half.qty == d_full.qty // 2
    assert d_over.qty == d_full.qty


def test_short_side_and_bad_levels(engine, state) -> None:
    d = engine.evaluate(proposal(side="short", entry=100.0, stop=101.0, target=98.0), state)
    assert d.approved and d.rr_ratio == pytest.approx(2.0)
    d_bad = engine.evaluate(proposal(stop=101.0), state)  # long with stop above entry
    assert d_bad.rejection is Rejection.BAD_LEVELS


def test_position_value_cap() -> None:
    # tight stop => huge qty by risk budget; the 20% position-value cap must bind
    eng = RiskEngine(RiskLimits(risk_per_trade_pct=1.0, max_position_pct=20.0))
    st = PortfolioState(equity=100_000.0, day_start_equity=100_000.0)
    d = eng.evaluate(proposal(), st)
    assert d.approved
    assert d.qty == 200  # 20% of 100k / $100 per share
    assert any("max position value" in n for n in d.notes)


def test_atr_levels() -> None:
    stop, target = atr_levels(100.0, 2.0, "long", 1.0, 2.0)
    assert (stop, target) == (98.0, 104.0)
    stop_s, target_s = atr_levels(100.0, 2.0, "short", 1.0, 2.0)
    assert (stop_s, target_s) == (102.0, 96.0)


def test_ewma_and_lgbm_vol() -> None:
    rng = np.random.default_rng(0)
    calm = rng.standard_normal(500) * 0.005
    wild = rng.standard_normal(500) * 0.03
    rets = np.concatenate([calm, wild])
    sigma = ewma_volatility(rets)
    assert sigma[900] > sigma[400] * 2  # detects the vol jump

    f = LightGBMVolForecaster(lags=10).fit(rets)
    assert f.predict_next(rets) > 0

    assert vol_scalar(forecast_vol=0.04, target_vol=0.01) == pytest.approx(0.25)
    assert vol_scalar(forecast_vol=0.005, target_vol=0.01) == 1.0  # can't lever above 1


def test_metalabeler_gates_bad_signals() -> None:
    rng = np.random.default_rng(1)
    n = 4000
    X = rng.standard_normal((n, 4))
    signal = np.where(rng.random(n) > 0.3, 1, 0)
    outcome = np.where(X[:, 0] > 0, 1, -1)  # success driven by feature 0
    ml = MetaLabeler(threshold=0.5).fit(X, signal, outcome)
    good = ml.size_multiplier(np.array([[2.0, 0, 0, 0]]))[0]
    bad = ml.size_multiplier(np.array([[-2.0, 0, 0, 0]]))[0]
    assert good > 0.6
    assert bad == 0.0


def _make_signals(n: int, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        win = rng.random() < 0.45
        out.append(
            dict(
                conf=rng.random(),
                vol=0.02,
                regime=0,
                entry=100.0,
                stop=99.0,
                target=102.0,
                outcome_r=2.0 if win else -1.0,
            )
        )
    return out


def test_sizing_env_and_baseline() -> None:
    limits = RiskLimits()
    signals = _make_signals(50)
    env = SizingEnv(signals, limits)
    obs, _ = env.reset()
    assert obs.shape == (6,)
    total_reward = 0.0
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(np.array([0.5]))
        total_reward += r
        done = term or trunc
    final = fixed_fraction_baseline(signals, limits)
    assert final > 0
    # zero-sizing agent ends flat:
    env2 = SizingEnv(signals, limits)
    env2.reset()
    done = False
    while not done:
        _, _, term, trunc, _ = env2.step(np.array([0.0]))
        done = term or trunc
    assert env2._state.equity == pytest.approx(100_000.0)
