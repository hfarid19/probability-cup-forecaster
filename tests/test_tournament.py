"""Tests for the bracket solver and the Monte-Carlo tournament simulator (offline)."""
from __future__ import annotations

import numpy as np

from wc_trader.backtest.bracket import advance_prob_from_3way, solve_bracket
from wc_trader.backtest.tournament import (
    KOMatch,
    TournamentSpec,
    _assign_thirds,
    simulate,
)

# ---- exact 8-team bracket ----

def test_solve_bracket_equal_teams():
    pairs = [("A", "B"), ("C", "D"), ("E", "F"), ("G", "H")]
    fc = solve_bracket(pairs, lambda a, b: 0.5)
    assert all(abs(p - 0.125) < 1e-12 for p in fc.p_champion.values())
    assert abs(sum(fc.p_champion.values()) - 1.0) < 1e-12
    assert abs(sum(fc.p_final.values()) - 2.0) < 1e-12       # two finalists


def test_solve_bracket_dominant_team():
    pairs = [("Star", "B"), ("C", "D"), ("E", "F"), ("G", "H")]
    fc = solve_bracket(pairs, lambda a, b: 0.95 if a == "Star" else (0.05 if b == "Star" else 0.5))
    assert fc.p_champion["Star"] > 0.8
    assert fc.p_qf["Star"] == 0.95


def test_advance_prob():
    assert advance_prob_from_3way(0.5, 0.3) == 0.65


# ---- third-place slot assignment ----

def test_assign_thirds_respects_constraints():
    slots = [(74, "3AB"), (77, "3BC")]
    qualified = {"A": "TeamA", "B": "TeamB", "C": "TeamC"}
    got = _assign_thirds(slots, qualified)
    assert set(got) == {74, 77}
    # 74 must take A or B; 77 must take B or C; no team used twice
    assert len(set(got.values())) == 2


def test_assign_thirds_forced_order():
    # slot 1 only accepts A; slot 2 accepts A or B -> backtracking must give B to slot 2
    slots = [(1, "3A"), (2, "3AB")]
    qualified = {"A": "TeamA", "B": "TeamB"}
    got = _assign_thirds(slots, qualified)
    assert got[1] == "TeamA" and got[2] == "TeamB"


# ---- Monte-Carlo simulator ----

def _mini_spec() -> TournamentSpec:
    groups = {"A": ["Strong", "a2", "a3", "a4"], "B": ["b1", "b2", "b3", "b4"]}
    fixtures = [(g, t1, t2) for g, ts in groups.items()
                for i, t1 in enumerate(ts) for t2 in ts[i + 1:]]
    ko = [KOMatch(13, "1A", "2B"), KOMatch(14, "1B", "2A"), KOMatch(15, "W13", "W14")]
    return TournamentSpec(groups=groups, fixtures=fixtures, ko=ko, final_number=15)


def _dominant_sampler(strong: str):
    def sample(h, a, rng: np.random.RandomState):
        if h == strong:
            return 3, 0
        if a == strong:
            return 0, 3
        return int(rng.poisson(1.2)), int(rng.poisson(1.2))
    return sample


def test_simulate_dominant_team_always_wins():
    spec = _mini_spec()
    res, finals = simulate(spec, _dominant_sampler("Strong"), n_sims=300, seed=1)
    assert res["Strong"]["champion"] == 1.0
    assert abs(sum(t["champion"] for t in res.values()) - 1.0) < 1e-9
    assert abs(sum(t["final"] for t in res.values()) - 2.0) < 1e-9
    assert all("Strong" in pair for pair in finals)          # always in the final
    assert abs(sum(finals.values()) - 1.0) < 1e-9


def test_simulate_symmetric_teams_roughly_uniform():
    spec = _mini_spec()
    def coin(h, a, rng):
        return int(rng.poisson(1.1)), int(rng.poisson(1.1))
    res, _ = simulate(spec, coin, n_sims=4000, seed=7)
    for t, r in res.items():
        assert 0.04 < r["champion"] < 0.22   # ~1/8 each, wide MC tolerance
