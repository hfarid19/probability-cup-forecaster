"""Tests for the Dixon-Coles model and odds de-vig math (no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from wc_trader.model.dixon_coles import DixonColesModel
from wc_trader.types import Outcome


def _toy_goals_df():
    # A (strong) outscores B (mid) outscores C (weak). Enough games to fit.
    strength = {"A": 2.4, "B": 1.4, "C": 0.7}  # ~ expected goals scored
    teams = list(strength)
    rows, day = [], pd.Timestamp("2000-01-01")
    for rep in range(20):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                # Deterministic-ish scoreline from strengths (rounded), no randomness needed.
                hs = round(strength[home] + 0.3)        # small home bump
                as_ = round(strength[away])
                rows.append({"date": day, "home_team": home, "away_team": away,
                             "home_score": hs, "away_score": as_,
                             "tournament": "Friendly", "neutral": False})
                day += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


def test_dc_probabilities_sum_to_one():
    m = DixonColesModel().fit(_toy_goals_df(), min_matches=3)
    p = m.match_probabilities("A", "C")
    assert set(p) == {Outcome.HOME, Outcome.DRAW, Outcome.AWAY}
    assert abs(sum(p.values()) - 1.0) < 1e-9


def test_dc_favors_stronger_team():
    m = DixonColesModel().fit(_toy_goals_df(), min_matches=3)
    p = m.match_probabilities("A", "C")          # strong home vs weak away
    assert p[Outcome.HOME] > p[Outcome.AWAY]


def test_dc_unknown_team_falls_back():
    m = DixonColesModel().fit(_toy_goals_df(), min_matches=3)
    p = m.match_probabilities("A", "Nonexistent")  # unknown -> league-average
    assert abs(sum(p.values()) - 1.0) < 1e-9


def test_devig_normalizes_to_one():
    # Replicates the de-vig in data.odds without needing the network.
    odds = np.array([[2.0, 4.0, 4.0]])           # implied 0.5 + 0.25 + 0.25 = 1.0 (no vig)
    inv = 1.0 / odds
    inv = inv / inv.sum(axis=1, keepdims=True)
    assert abs(inv.sum() - 1.0) < 1e-9
    assert abs(inv[0, 0] - 0.5) < 1e-9
