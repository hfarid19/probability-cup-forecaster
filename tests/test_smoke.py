"""Smoke tests for the Milestone 1 skeleton."""
from __future__ import annotations

from wc_trader.config import load_config
from wc_trader.model.elo import EloModel
from wc_trader.strategy.value import ValueStrategy, kelly_stake
from wc_trader.types import Outcome


def test_elo_probabilities_sum_to_one():
    probs = EloModel().match_probabilities("A", "B")
    assert set(probs) == {Outcome.HOME, Outcome.DRAW, Outcome.AWAY}
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_elo_home_advantage_helps_equal_teams():
    probs = EloModel(ratings={"A": 1500, "B": 1500}).match_probabilities("A", "B")
    assert probs[Outcome.HOME] > probs[Outcome.AWAY]


def test_kelly_zero_when_no_edge():
    # prob == implied (1/2.0 = 0.5): no edge -> zero stake
    assert kelly_stake(0.5, 2.0, 1000, 0.25) == 0.0


def test_kelly_positive_with_edge():
    assert kelly_stake(0.6, 2.0, 1000, 0.25) > 0.0


def test_strategy_backs_underpriced_outcome():
    from wc_trader.types import OrderBook, PriceLevel, Side

    cfg = load_config()
    strat = ValueStrategy(cfg)
    # market implies ~40% (back odds 2.5); model says 55% -> should BACK
    book = OrderBook(1, backs=[PriceLevel(2.5, 100)], lays=[PriceLevel(2.56, 100)])
    sig = strat.evaluate(1, model_prob=0.55, book=book, bankroll=1000)
    assert sig is not None and sig.side == Side.BACK
