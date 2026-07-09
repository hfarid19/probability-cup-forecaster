"""Offline tests for the lineup/form news layer (synthetic lineups, no network)."""
from __future__ import annotations

import numpy as np

from wc_trader.model.lineup_adjust import (
    NewsAdjustedDC,
    NewsCoefficients,
    NewsFeatureBuilder,
    TeamMatchNews,
    adjusted_lambdas,
    fit_coefficients,
)
from wc_trader.types import Outcome


class _FlatDC:
    """Stand-in DC: independent Poisson(1.3, 1.3) grid for any pairing."""
    def score_grid(self, home, away, neutral=True):
        from scipy.stats import poisson
        gv = np.arange(11)
        g = np.outer(poisson.pmf(gv, 1.3), poisson.pmf(gv, 1.3))
        return g / g.sum()


def _rec(num, date, home, away, sh, sa, goals_h=(), goals_a=()):
    def side(team, starters, goals):
        return {"team": team,
                "players": [{"id": p, "name": p, "starter": True, "position": 1}
                            for p in starters],
                "goals": [{"player": g, "assist": None, "minute": 10} for g in goals]}
    return {"match_number": num, "date": date, "home": side(home, sh, goals_h),
            "away": side(away, sa, goals_a)}


XI_A = [f"a{i}" for i in range(11)]
XI_A_ROTATED = [f"a{i}" for i in range(6, 17)]   # 5 keep, 6 new
XI_B = [f"b{i}" for i in range(11)]


def test_rotation_delta_settled_vs_rotated():
    recs = [
        _rec(1, "2026-06-12", "A", "B", XI_A, XI_B),
        _rec(2, "2026-06-16", "A", "B", XI_A, XI_B),          # same XI -> rotation 0
        _rec(3, "2026-06-20", "A", "B", XI_A_ROTATED, XI_B),  # 6 of 11 new
    ]
    b = NewsFeatureBuilder(_FlatDC(), recs)
    _, f2 = b.per_record[2]      # third match's own features (no pair-collision risk)
    # third match: A started 5 of 11 players who started both prior matches
    assert abs(f2["A"].rotation_delta - (1 - 5 / 11)) < 1e-9
    assert f2["B"].rotation_delta == 0.0                      # B unchanged
    # date-aware lookup returns the right occurrence of a repeated pairing
    early = b.news_for("A", "B", date="2026-06-12")
    late = b.news_for("A", "B", date="2026-06-20")
    assert early[0].rotation_delta == 0.0 and late[0].rotation_delta > 0.4
    # ambiguous lookup without a date must fail loudly, not leak a future match
    import pytest
    with pytest.raises(ValueError):
        b.news_for("A", "B")


def test_hot_delta_positive_when_scorers_present():
    # A scores 3 with the same XI (expected 1.3): over-performance carried by starters.
    recs = [
        _rec(1, "2026-06-12", "A", "B", XI_A, XI_B, goals_h=("a1", "a1", "a2")),
        _rec(2, "2026-06-16", "A", "B", XI_A, XI_B),
    ]
    b = NewsFeatureBuilder(_FlatDC(), recs)
    _, f = b.per_record[1]
    assert f["A"].hot_delta > 1.0        # (3 − 1.3)/1 × presence 1.0
    assert f["B"].hot_delta < 0.0        # scored 0, expected 1.3


def test_adjusted_lambdas_directions():
    c = NewsCoefficients(beta_own=0.5, beta_opp=0.3, gamma=0.2)
    rested = TeamMatchNews(rotation_delta=0.6, hot_delta=0.0)
    normal = TeamMatchNews(rotation_delta=0.0, hot_delta=0.0)
    lh, la = adjusted_lambdas(1.5, 1.2, rested, normal, c)
    assert lh < 1.5                       # own rotation weakens own attack
    assert la > 1.2                       # opponent rotation weakens their defense
    hot = TeamMatchNews(rotation_delta=0.0, hot_delta=1.0)
    lh2, _ = adjusted_lambdas(1.5, 1.2, hot, normal, c)
    assert lh2 > 1.5                      # form boost


def test_fit_recovers_rotation_effect():
    rng = np.random.RandomState(0)
    true = NewsCoefficients(beta_own=0.6, beta_opp=0.2, gamma=0.0)
    samples = []
    for _ in range(800):
        news_h = TeamMatchNews(rng.uniform(0, 0.8), 0.0)
        news_a = TeamMatchNews(rng.uniform(0, 0.8), 0.0)
        lh, la = adjusted_lambdas(1.4, 1.4, news_h, news_a, true)
        samples.append({"lam_h": 1.4, "lam_a": 1.4, "news_h": news_h, "news_a": news_a,
                        "goals_h": rng.poisson(lh), "goals_a": rng.poisson(la)})
    est = fit_coefficients(samples)
    assert abs(est.beta_own - 0.6) < 0.2
    assert abs(est.beta_opp - 0.2) < 0.2


def test_news_adjusted_dc_shifts_probabilities():
    recs = [
        _rec(1, "2026-06-12", "A", "B", XI_A, XI_B),
        _rec(2, "2026-06-16", "A", "B", XI_A_ROTATED, XI_B),   # A rotates heavily
    ]
    dc = _FlatDC()
    b = NewsFeatureBuilder(dc, recs)
    coef = NewsCoefficients(beta_own=0.8, beta_opp=0.0, gamma=0.0)
    adj = NewsAdjustedDC(dc, b, coef)
    p_adj = adj.match_probabilities("A", "B", date="2026-06-16")
    assert p_adj[Outcome.HOME] < 1/3 - 0.02   # rotated A now underdog vs symmetric base
    assert abs(sum(p_adj.values()) - 1.0) < 1e-9
