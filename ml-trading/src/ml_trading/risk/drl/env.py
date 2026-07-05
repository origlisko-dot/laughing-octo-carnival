"""DRL position-sizing environment (advanced stage).

The agent chooses *how much* of the risk budget to use for each pre-approved
signal — never whether limits apply. Its action in [0, 1] becomes the
`size_multiplier` on a `TradeProposal`, so the deterministic `RiskEngine` still
bounds everything (R/R filter, exposure caps, kill switch).

Implements the Gymnasium API when the `drl` extra is installed, but the class
also works standalone (duck-typed) for tests without gymnasium. Train with
PPO/SAC via stable-baselines3; the acceptance gate is beating a fixed-fraction
baseline out-of-sample, otherwise the DRL sizer is dropped.
"""

from __future__ import annotations

import numpy as np

from ml_trading.config import RiskLimits
from ml_trading.risk.engine import PortfolioState, RiskEngine, TradeProposal

try:
    import gymnasium
    from gymnasium import spaces
except ImportError:  # optional extra
    gymnasium = None
    spaces = None

_BASE = gymnasium.Env if gymnasium is not None else object

OBS_DIM = 6  # [signal_conf, forecast_vol, regime, drawdown, loss_streak, exposure]


class SizingEnv(_BASE):
    """Episode = a sequence of historical signals with known outcomes.

    signals: array of dicts with keys
        conf (0..1), vol (forecast), regime (int), entry, stop, target, outcome_r
        where outcome_r is the realized R-multiple of the trade (+2.0, -1.0, ...).
    """

    metadata = {"render_modes": []}

    def __init__(self, signals: list[dict], limits: RiskLimits, initial_equity: float = 100_000.0):
        if gymnasium is not None:
            super().__init__()
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,))
            self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,))
        self.signals = signals
        self.engine = RiskEngine(limits)
        self.initial_equity = initial_equity
        self._i = 0
        self._state = PortfolioState(equity=initial_equity, day_start_equity=initial_equity)

    def _obs(self) -> np.ndarray:
        s = self.signals[min(self._i, len(self.signals) - 1)]
        dd = (self._state.day_start_equity - self._state.equity) / self._state.day_start_equity
        return np.array(
            [
                s["conf"],
                s["vol"],
                float(s.get("regime", 0)),
                dd,
                float(self._state.consecutive_losses),
                len(self._state.open_positions) / max(self.engine.limits.max_open_positions, 1),
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if gymnasium is not None:
            super().reset(seed=seed)
        self._i = 0
        self._state = PortfolioState(
            equity=self.initial_equity, day_start_equity=self.initial_equity
        )
        return self._obs(), {}

    def step(self, action):
        mult = float(np.clip(np.asarray(action).ravel()[0], 0.0, 1.0))
        s = self.signals[self._i]
        proposal = TradeProposal(
            ticker=s.get("ticker", "X"),
            side=s.get("side", "long"),
            entry=s["entry"],
            stop=s["stop"],
            target=s["target"],
            size_multiplier=mult,
        )
        decision = self.engine.evaluate(proposal, self._state)

        pnl = 0.0
        if decision.approved:
            pnl = decision.risk_amount * s["outcome_r"]
            self._state.equity += pnl
            self._state.consecutive_losses = 0 if pnl > 0 else self._state.consecutive_losses + 1

        # Reward: log equity growth (risk-sensitive; compounding-aware).
        prev = self._state.equity - pnl
        reward = float(np.log(max(self._state.equity, 1e-6) / max(prev, 1e-6)))

        self._i += 1
        terminated = self._i >= len(self.signals)
        truncated = self.engine.kill_switch_active(self._state)
        return self._obs(), reward, terminated, truncated, {"pnl": pnl, "approved": decision.approved}


def fixed_fraction_baseline(signals: list[dict], limits: RiskLimits, equity: float = 100_000.0) -> float:
    """Final equity of always using the full deterministic size (multiplier=1).

    A trained DRL sizer must beat this out-of-sample or it is rejected.
    """
    env = SizingEnv(signals, limits, equity)
    env.reset()
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(np.array([1.0]))
        done = term or trunc
    return env._state.equity
