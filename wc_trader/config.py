"""Typed configuration: config.yaml for parameters, .env for secrets."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class StrategyConfig(BaseModel):
    edge_threshold: float = 0.04
    kelly_fraction: float = 0.25


class RiskConfig(BaseModel):
    max_position_per_market: float = 0.05
    max_exposure_per_team: float = 0.15
    max_total_deployed: float = 0.50
    max_drawdown_halt: float = 0.20


class BetfairConfig(BaseModel):
    commission_rate: float = 0.05
    markets: list[str] = Field(default_factory=lambda: ["MATCH_ODDS"])
    poll_interval_seconds: int = 30


class AppConfig(BaseModel):
    bankroll: float = 1000.0
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    betfair: BetfairConfig = Field(default_factory=BetfairConfig)


class Secrets(BaseSettings):
    """Loaded from environment / .env — never hard-coded or committed."""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    betfair_username: str | None = None
    betfair_password: str | None = None
    betfair_app_key: str | None = None
    betfair_cert_path: str | None = None
    betfair_key_path: str | None = None


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    p = Path(path)
    if p.exists():
        data = yaml.safe_load(p.read_text()) or {}
        return AppConfig(**data)
    return AppConfig()
