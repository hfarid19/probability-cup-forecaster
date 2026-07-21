"""Tests for the Probability Cup forecast math (no network, no API key)."""
from __future__ import annotations

import numpy as np
import pytest

from probability_cup import forecast


def test_devig_1x2_sums_to_one_and_removes_margin():
    p = forecast.devig_1x2(2.0, 3.5, 4.0)  # ~1.06 book overround
    assert abs(sum(p.values()) - 1.0) < 1e-9
    # favourite still favoured, all strictly between 0 and 1
    assert p["HOME"] > p["AWAY"] > 0
    assert all(0 < v < 1 for v in p.values())


def test_american_and_two_way_devig():
    assert abs(forecast.american_to_prob(-110) - 0.5238) < 1e-3
    assert abs(forecast.american_to_prob(+150) - 0.4) < 1e-3
    # a -110/-110 two-way market is a fair coin flip after de-vig
    yes = forecast.american_to_prob(-110)
    assert abs(forecast.devig_two_way(yes, yes) - 0.5) < 1e-9


def test_implied_lambdas_reproduce_targets():
    rho = -0.1
    devigged = {"HOME": 0.55, "DRAW": 0.25, "AWAY": 0.20}
    lam = forecast.implied_lambdas(devigged, prob_under_2_5=0.50, rho=rho)
    s = forecast._grid_summary(forecast.score_grid(*lam, rho))
    assert abs(s["home"] - 0.55) < 0.02
    assert abs(s["under25"] - 0.50) < 0.02
    assert lam[0] > lam[1]  # home favoured -> higher expected goals


def test_score_grid_normalised():
    g = forecast.score_grid(1.6, 1.1, -0.1)
    assert abs(g.sum() - 1.0) < 1e-9
    assert (g >= 0).all()


def test_result_probs_sum_to_one():
    r = forecast.prob_result((1.5, 1.0), -0.1)
    assert abs(sum(r.values()) - 1.0) < 1e-9


def test_timing_windows_monotonic_in_goals():
    # more expected goals -> higher chance of a goal in any fixed window
    low = forecast.prob_late_goal((0.6, 0.5))
    high = forecast.prob_late_goal((1.8, 1.4))
    assert 0 < low < high < 1


def test_boldness_amplifies_away_from_crowd_with_clamps():
    # fair below crowd -> bold pushed further below, respecting a floor
    assert forecast.apply_boldness(fair=50, crowd=62, factor=1.5) == 44
    assert forecast.apply_boldness(fair=25, crowd=45, factor=2.0, floor=18) == 18
    # never leaves [1, 99]
    assert forecast.apply_boldness(fair=2, crowd=40, factor=5.0) >= 1
    assert forecast.apply_boldness(fair=98, crowd=60, factor=5.0) <= 99


def test_client_requires_key(monkeypatch):
    from probability_cup.client import SportsPredictClient

    monkeypatch.delenv("SPORTSPREDICT_KEY", raising=False)
    with pytest.raises(RuntimeError, match="No API key"):
        SportsPredictClient()
