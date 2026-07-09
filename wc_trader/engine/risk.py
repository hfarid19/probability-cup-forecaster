"""Risk manager: hard caps + drawdown kill-switch. Fails closed (rejects on doubt)."""
from __future__ import annotations

from loguru import logger

from ..config import AppConfig
from ..strategy.value import Signal


class RiskManager:
    def __init__(self, cfg: AppConfig, bankroll: float):
        self.cfg = cfg
        self.bankroll = bankroll
        self.peak_bankroll = bankroll
        self.deployed = 0.0
        self.team_exposure: dict[int, float] = {}
        self.halted = False

    def update_bankroll(self, value: float) -> None:
        self.bankroll = value
        self.peak_bankroll = max(self.peak_bankroll, value)
        drawdown = 1 - value / self.peak_bankroll if self.peak_bankroll else 0.0
        if drawdown >= self.cfg.risk.max_drawdown_halt:
            self.halted = True
            logger.error(f"[RISK] drawdown {drawdown:.1%} >= halt "
                         f"{self.cfg.risk.max_drawdown_halt:.0%} — HALTING ALL TRADING")

    def clamp(self, signal: Signal) -> Signal | None:
        """Apply caps; return an adjusted signal or None if it can't be sized safely."""
        if self.halted:
            logger.warning("[RISK] halted — rejecting signal")
            return None

        r = self.cfg.risk
        stake = min(signal.stake, r.max_position_per_market * self.bankroll)

        team_cap = r.max_exposure_per_team * self.bankroll
        used = self.team_exposure.get(signal.selection_id, 0.0)
        stake = min(stake, max(0.0, team_cap - used))

        total_cap = r.max_total_deployed * self.bankroll
        stake = min(stake, max(0.0, total_cap - self.deployed))

        if stake <= 0:
            return None
        signal.stake = stake
        return signal

    def register_fill(self, signal: Signal) -> None:
        self.deployed += signal.stake
        self.team_exposure[signal.selection_id] = (
            self.team_exposure.get(signal.selection_id, 0.0) + signal.stake
        )
