"""Central configuration.

All secrets come from environment variables only; YAML files hold
non-secret research configuration (universe, intervals, risk limits).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

INTERVALS: tuple[str, ...] = ("1m", "5m", "15m", "1h", "1d", "1w", "1M")

INTERVAL_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3_600,
    "1d": 86_400,
    "1w": 604_800,
    "1M": 2_592_000,  # nominal 30d; monthly bars are calendar-aligned, this is only for ordering
}


class Secrets(BaseSettings):
    """API credentials, loaded from the environment only."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    finnhub_api_key: str = ""
    polygon_api_key: str = ""


class RiskLimits(BaseModel):
    """Hard limits enforced by the deterministic risk core. ML may never relax these."""

    risk_per_trade_pct: float = Field(0.75, gt=0, le=5.0, description="% of equity risked per trade")
    min_risk_reward: float = Field(2.0, ge=1.0, description="minimum take-profit / stop distance ratio")
    max_open_positions: int = Field(5, ge=1)
    max_sector_exposure_pct: float = Field(30.0, gt=0, le=100.0)
    max_daily_loss_pct: float = Field(3.0, gt=0, description="kill switch: halt trading for the day")
    max_position_pct: float = Field(20.0, gt=0, le=100.0, description="max % of equity in one position")
    loss_streak_cooldown: int = Field(3, ge=1, description="consecutive losses before size reduction")
    loss_streak_size_factor: float = Field(0.5, gt=0, le=1.0)
    kelly_cap_fraction: float = Field(0.5, gt=0, le=1.0, description="fractional Kelly ceiling")


class DataConfig(BaseModel):
    root: Path = Path("data")
    intervals: list[str] = list(INTERVALS)
    universe: list[str] = ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "AMZN", "GOOGL", "META"]
    sectors: dict[str, str] = {}


class LabelConfig(BaseModel):
    profit_atr_mult: float = 2.0
    stop_atr_mult: float = 1.0
    max_holding_bars: int = 60
    atr_window: int = 14


class CVConfig(BaseModel):
    n_splits: int = 5
    embargo_bars: int = 60
    purge_bars: int = 60


class BacktestConfig(BaseModel):
    commission_per_share: float = 0.005
    min_commission: float = 1.0
    slippage_bps: float = 2.0
    initial_equity: float = 100_000.0


class AppConfig(BaseModel):
    data: DataConfig = DataConfig()
    labels: LabelConfig = LabelConfig()
    cv: CVConfig = CVConfig()
    risk: RiskLimits = RiskLimits()
    backtest: BacktestConfig = BacktestConfig()

    @classmethod
    def load(cls, path: str | Path | None = None) -> AppConfig:
        if path is None:
            return cls()
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls.model_validate(raw)
