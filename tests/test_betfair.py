"""Tests for BetfairVenue's pure mapping/rounding logic (no live account)."""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace as NS

from wc_trader.adapters.betfair import (
    catalogue_to_market,
    infer_outcome,
    market_book_to_orderbooks,
    round_to_tick,
)
from wc_trader.types import Outcome


def test_round_to_tick_bands():
    assert round_to_tick(3.14) == 3.15     # 3-4 band, 0.05 tick
    assert round_to_tick(1.014) == 1.01    # 1-2 band, 0.01 tick
    assert round_to_tick(25.4) == 25.0     # 20-30 band, 1.0 tick
    assert round_to_tick(2.00) == 2.00     # band boundary is valid
    assert round_to_tick(6.7) in (6.6, 6.8)  # 6-10 band, 0.2 tick (tie)


def test_round_to_tick_clamps():
    assert round_to_tick(0.5) == 1.01      # below min odds
    assert round_to_tick(5000) == 1000.0   # above max odds


def test_infer_outcome():
    assert infer_outcome("England", "England v France") == Outcome.HOME
    assert infer_outcome("France", "England v France") == Outcome.AWAY
    assert infer_outcome("The Draw", "England v France") == Outcome.DRAW
    assert infer_outcome("Somewhere", "A vs B") is None


def test_catalogue_to_market():
    cat = NS(
        market_id="1.234",
        market_name="Match Odds",
        market_start_time=datetime(2026, 7, 5, 18, 0),
        event=NS(name="Brazil v Argentina"),
        runners=[NS(selection_id=47, runner_name="Brazil"),
                 NS(selection_id=48, runner_name="Argentina"),
                 NS(selection_id=58805, runner_name="The Draw")],
    )
    m = catalogue_to_market(cat)
    assert m.market_id == "1.234"
    assert m.event_name == "Brazil v Argentina"
    outcomes = {r.name: r.outcome for r in m.runners}
    assert outcomes["Brazil"] == Outcome.HOME
    assert outcomes["Argentina"] == Outcome.AWAY
    assert outcomes["The Draw"] == Outcome.DRAW
    assert m.start_time.startswith("2026-07-05")


def test_market_book_to_orderbooks():
    book = NS(runners=[
        NS(selection_id=47, ex=NS(
            available_to_back=[NS(price=2.5, size=100.0), NS(price=2.48, size=50.0)],
            available_to_lay=[NS(price=2.54, size=80.0)])),
        NS(selection_id=48, ex=NS(available_to_back=[], available_to_lay=[])),
    ])
    obs = market_book_to_orderbooks(book)
    assert obs[47].best_back == 2.5
    assert obs[47].best_lay == 2.54
    assert obs[47].backs[1].size == 50.0
    assert obs[48].best_back is None
