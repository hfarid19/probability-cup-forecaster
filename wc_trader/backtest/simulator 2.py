"""Backtest harness (Milestone 3 — STUB).

Replay Betfair historical order-book data against actual results: for each market, ask
the model for probabilities, run the strategy, simulate fills against the recorded book,
apply commission + slippage, and report edge, calibration (log-loss / Brier / reliability
curve) and PnL. The key gate: the model must beat the closing line out-of-sample.
"""
from __future__ import annotations

from ..config import AppConfig


class Backtester:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def run(self, *args, **kwargs):
        raise NotImplementedError("M3: replay Betfair historical data, score calibration + PnL")
