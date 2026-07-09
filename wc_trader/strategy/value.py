"""Value strategy: compare model probability to market implied probability, decide
back vs lay, and size with fractional Kelly.

  - BACK when the model thinks an outcome is UNDERPRICED (model_prob > implied @ best back)
  - LAY  when the model thinks an outcome is OVERPRICED  (model_prob < implied @ best lay)

The edge threshold (config) is set to cover Betfair commission + expected slippage, so a
signal only fires when there's edge left after costs.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import AppConfig
from ..types import OrderBook, Side


@dataclass
class Signal:
    selection_id: int
    side: Side
    price: float          # decimal odds to take
    stake: float          # recommended stake (before risk clamp)
    edge: float           # probability edge after threshold
    model_prob: float
    market_prob: float


def implied_prob(odds: float) -> float:
    """Implied probability from decimal odds (ignores overround)."""
    return 1.0 / odds if odds > 0 else 0.0


def kelly_stake(prob: float, odds: float, bankroll: float, fraction: float) -> float:
    """Fractional Kelly for decimal odds. b = odds - 1; f* = (p*b - (1-p)) / b."""
    b = odds - 1.0
    if b <= 0:
        return 0.0
    f = (prob * b - (1 - prob)) / b
    return max(0.0, f) * fraction * bankroll


class ValueStrategy:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def evaluate(self, selection_id: int, model_prob: float, book: OrderBook,
                 bankroll: float) -> Signal | None:
        thr = self.cfg.strategy.edge_threshold
        frac = self.cfg.strategy.kelly_fraction
        back_odds, lay_odds = book.best_back, book.best_lay
        if back_odds is None or lay_odds is None:
            return None

        edge_back = model_prob - implied_prob(back_odds)
        edge_lay = implied_prob(lay_odds) - model_prob

        # Prefer whichever side has the larger qualifying edge.
        if edge_back >= thr and edge_back >= edge_lay:
            stake = kelly_stake(model_prob, back_odds, bankroll, frac)
            return Signal(selection_id, Side.BACK, back_odds, stake, edge_back,
                          model_prob, implied_prob(back_odds))

        if edge_lay >= thr:
            # Laying at odds L == backing the complement at L/(L-1); size Kelly on that.
            comp_odds = lay_odds / (lay_odds - 1.0)
            stake = kelly_stake(1 - model_prob, comp_odds, bankroll, frac)
            return Signal(selection_id, Side.LAY, lay_odds, stake, edge_lay,
                          model_prob, implied_prob(lay_odds))

        return None
