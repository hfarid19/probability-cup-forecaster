"""Tests for M2/M3: metrics correctness and the walk-forward backtester."""
from __future__ import annotations

import math

import pandas as pd

from wc_trader.backtest.metrics import (
    accuracy,
    base_rate_baseline,
    brier_score,
    log_loss,
    reliability_curve,
)
from wc_trader.backtest.simulator import CLASSES, match_outcome, run_elo_backtest


def test_match_outcome():
    assert match_outcome(2, 1) == "HOME"
    assert match_outcome(0, 0) == "DRAW"
    assert match_outcome(1, 3) == "AWAY"


def test_log_loss_perfect_is_zero():
    probs = [{"HOME": 1.0, "DRAW": 0.0, "AWAY": 0.0}]
    assert log_loss(probs, ["HOME"]) < 1e-9


def test_log_loss_uniform():
    probs = [{"HOME": 1 / 3, "DRAW": 1 / 3, "AWAY": 1 / 3}]
    assert abs(log_loss(probs, ["HOME"]) - math.log(3)) < 1e-9


def test_brier_and_accuracy():
    probs = [{"HOME": 0.7, "DRAW": 0.2, "AWAY": 0.1}]
    assert accuracy(probs, ["HOME"]) == 1.0
    assert accuracy(probs, ["AWAY"]) == 0.0
    # Brier for this single confident-correct case
    expected = (0.7 - 1) ** 2 + 0.2**2 + 0.1**2
    assert abs(brier_score(probs, ["HOME"], CLASSES) - expected) < 1e-9


def test_base_rate_sums_to_one():
    base = base_rate_baseline(["HOME", "HOME", "DRAW", "AWAY"], CLASSES)
    assert abs(sum(base.values()) - 1.0) < 1e-9
    assert base["HOME"] == 0.5


def test_reliability_curve_bins():
    probs = [{"HOME": 0.95, "DRAW": 0.03, "AWAY": 0.02}] * 10
    bins = reliability_curve(probs, ["HOME"] * 10, CLASSES, n_bins=10)
    assert any(b.lo == 0.9 for b in bins)


def _toy_df():
    # Five teams of distinct strength; the stronger team wins. A single global base rate
    # can't capture this (it ignores WHO is playing), so a model that learns per-team
    # strength should beat the base-rate predictor — which is the whole point of Elo.
    strength = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}
    teams = list(strength)
    rows, day = [], pd.Timestamp("2000-01-01")
    for _ in range(12):  # rounds
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                hs, as_ = (2, 0) if strength[home] > strength[away] else (0, 2)
                rows.append({"date": day, "home_team": home, "away_team": away,
                             "home_score": hs, "away_score": as_,
                             "tournament": "Friendly", "neutral": False})
                day += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


def test_backtest_runs_and_learns():
    df = _toy_df()
    report = run_elo_backtest(df, eval_start="2000-03-01")  # evaluate later games, after learning
    assert report.n_eval > 0
    assert 0.0 <= report.log_loss < math.log(3)        # better than uniform guessing
    # Having learned the strength ordering, the model should beat the base-rate predictor.
    assert report.skill_vs_baseline > 0
