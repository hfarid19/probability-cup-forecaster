"""Executor: turn risk-approved signals into venue orders; track fills + audit log."""
from __future__ import annotations

from loguru import logger

from ..adapters.base import Venue
from ..strategy.value import Signal
from .risk import RiskManager


class Executor:
    def __init__(self, venue: Venue, risk: RiskManager):
        self.venue = venue
        self.risk = risk

    def execute(self, signal: Signal) -> str | None:
        approved = self.risk.clamp(signal)
        if approved is None:
            return None
        order_id = self.venue.place_order(
            approved.selection_id, approved.side, approved.price, approved.stake
        )
        self.risk.register_fill(approved)
        logger.info(
            f"[EXEC] {approved.side.value} £{approved.stake:.2f} @ {approved.price:.2f} "
            f"(edge {approved.edge:+.1%}) order={order_id}"
        )
        return order_id
