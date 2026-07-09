"""Tests for the Groll et al. (2019) replication pieces (offline, no network)."""
from __future__ import annotations

import pandas as pd

from wc_trader.data.fifa_rank import RankLookup, normalize_team
from wc_trader.data.wc_meta import confederation, is_host, iso3
from wc_trader.data.worldbank import IndicatorLookup, parse_worldbank_json
from wc_trader.model.groll_rf import (
    HybridRF,
    TeamSnapshot,
    feature_columns,
    match_rows,
    poisson_outcome_probs,
)
from wc_trader.types import Outcome


def _snap(**kw) -> TeamSnapshot:
    base = dict(attack=0.0, defense=0.0, rank_points=1500.0, rank_pos=10.0,
                gdp_pc=30000.0, population=5e7, host=False, confed="UEFA")
    base.update(kw)
    return TeamSnapshot(**base)


# ---- metadata ----

def test_meta_lookups():
    assert confederation("Brazil") == "CONMEBOL"
    assert confederation("Atlantis") == "OTHER"
    assert iso3("England") == "GBR"          # UK constituents share GBR
    assert is_host("United States", 2026) and is_host("Mexico", 2026)
    assert not is_host("Brazil", 2026)


def test_rank_name_normalization():
    assert normalize_team("Korea Republic") == "South Korea"
    assert normalize_team("IR Iran") == "Iran"
    assert normalize_team("Eritrea (unranked)") == "Eritrea"
    assert normalize_team("Brazil") == "Brazil"


# ---- FIFA rank as-of ----

def test_rank_lookup_asof():
    df = pd.DataFrame({
        "team": ["A", "B", "A", "B"],
        "date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-06-01", "2020-06-01"]),
        "points": [1000.0, 900.0, 950.0, 990.0],
    })
    df["rank"] = df.groupby("date")["points"].rank(ascending=False, method="min").astype(int)
    lk = RankLookup(df)
    assert lk.table_asof("2020-03-01")["A"] == (1000.0, 1)   # first release
    assert lk.table_asof("2021-01-01")["A"] == (950.0, 2)    # latest release, B overtook
    assert lk.table_asof("2019-01-01") == {}                 # before any release
    assert lk.staleness_days("2020-07-01") == 30


# ---- FIFA live-API snapshot parser ----

def test_parse_live_rankings_uses_prev_points_only():
    from wc_trader.data.fifa_rank import parse_live_rankings
    payload = {"Results": [
        {"TeamName": [{"Locale": "en-GB", "Description": "Argentina"}],
         "Rank": 2, "TotalPoints": 1925.15,          # live (contaminated) — must be ignored
         "PrevRank": 1, "PrevPoints": 1877.27},      # official pre-WC release — used
        {"TeamName": [{"Locale": "en-GB", "Description": "Newland"}],
         "Rank": 50, "TotalPoints": 1400.0, "PrevRank": None, "PrevPoints": None},
    ]}
    rows = parse_live_rankings(payload, release_date="2026-06-11")
    assert rows == [{"team": "Argentina", "date": "2026-06-11", "points": 1877.27}]


# ---- World Bank ----

def test_worldbank_parse_and_asof():
    payload = [{"page": 1}, [
        {"countryiso3code": "BRA", "date": "2020", "value": 8000.0},
        {"countryiso3code": "BRA", "date": "2021", "value": None},
        {"countryiso3code": "BRA", "date": "2019", "value": 7500.0},
    ]]
    rows = parse_worldbank_json(payload)
    lk = IndicatorLookup(pd.DataFrame(rows))
    assert lk.value_asof("BRA", 2021) == 8000.0   # latest non-null <= 2021
    assert lk.value_asof("BRA", 2019) == 7500.0
    assert lk.value_asof("XXX", 2020) is None
    assert lk.value_asof(None, 2020) is None
    assert parse_worldbank_json([{"page": 1}, None]) == []


# ---- feature building ----

def test_match_rows_shape_and_symmetry():
    snaps = {"H": _snap(attack=0.5, host=True), "A": _snap(attack=-0.2, confed="CAF")}
    rows = match_rows("m1", "H", "A", 2, 1, snaps)
    assert len(rows) == 2
    r_home, r_away = rows
    assert r_home["goals"] == 2 and r_away["goals"] == 1
    assert r_home["t_attack"] == 0.5 and r_home["o_attack"] == -0.2
    assert r_away["t_attack"] == -0.2 and r_away["o_attack"] == 0.5   # mirrored
    assert r_home["t_host"] == 1.0 and r_away["o_host"] == 1.0
    assert r_away["t_confed_CAF"] == 1.0 and r_away["t_confed_UEFA"] == 0.0
    assert set(feature_columns()).issubset(r_home.keys())


# ---- double Poisson ----

def test_poisson_outcome_probs_valid_and_ordered():
    p = poisson_outcome_probs(2.0, 0.8)
    assert abs(sum(p.values()) - 1.0) < 1e-9
    assert p[Outcome.HOME] > p[Outcome.AWAY]
    sym = poisson_outcome_probs(1.3, 1.3)
    assert abs(sym[Outcome.HOME] - sym[Outcome.AWAY]) < 1e-9


# ---- RPS + likelihood (the paper's §4 metrics) ----

def test_rps_perfect_and_worst():
    from wc_trader.backtest.metrics import rps
    sure_home = [{"HOME": 1.0, "DRAW": 0.0, "AWAY": 0.0}]
    assert rps(sure_home, ["HOME"]) < 1e-12                  # perfect
    assert abs(rps(sure_home, ["AWAY"]) - 1.0) < 1e-12       # maximally wrong
    # a draw prediction is 'closer' to home than an away prediction is (ordering matters)
    sure_draw = [{"HOME": 0.0, "DRAW": 1.0, "AWAY": 0.0}]
    sure_away = [{"HOME": 0.0, "DRAW": 0.0, "AWAY": 1.0}]
    assert rps(sure_draw, ["HOME"]) < rps(sure_away, ["HOME"])


def test_avg_likelihood():
    from wc_trader.backtest.metrics import avg_likelihood
    probs = [{"HOME": 0.5, "DRAW": 0.3, "AWAY": 0.2}, {"HOME": 0.1, "DRAW": 0.2, "AWAY": 0.7}]
    assert abs(avg_likelihood(probs, ["HOME", "AWAY"]) - 0.6) < 1e-12


# ---- paired bootstrap ----

def test_paired_bootstrap_detects_clear_winner():
    from wc_trader.backtest.metrics import paired_logloss_bootstrap
    good = [{"HOME": 0.8, "DRAW": 0.1, "AWAY": 0.1}] * 50
    bad = [{"HOME": 0.34, "DRAW": 0.33, "AWAY": 0.33}] * 50
    outs = ["HOME"] * 50
    bs = paired_logloss_bootstrap(good, bad, outs)
    assert bs["mean_diff"] < 0            # A (good) better
    assert bs["ci_high"] < 0              # significantly so
    # identical models -> ~zero diff, CI straddles 0
    bs2 = paired_logloss_bootstrap(bad, bad, outs)
    assert bs2["mean_diff"] == 0.0
    assert bs2["ci_low"] <= 0 <= bs2["ci_high"]


# ---- end-to-end on synthetic data ----

def test_hybrid_rf_learns_attack_signal():
    # Strong-attack teams score more; the RF should recover that direction.
    snaps = {"S": _snap(attack=1.0), "W": _snap(attack=-1.0)}
    rows = []
    for i in range(60):
        rows.extend(match_rows(f"m{i}", "S", "W", 3, 0, snaps))
        rows.extend(match_rows(f"n{i}", "W", "S", 0, 3, snaps))
    train = pd.DataFrame(rows)
    rf = HybridRF(n_estimators=50).fit(train)

    pair = pd.DataFrame(match_rows("t", "S", "W", None, None, snaps))
    probs = rf.match_probabilities_from_rows(pair)
    assert probs[Outcome.HOME] > 0.6                      # strong side favored
    lam = rf.predict_lambda(pair)
    assert lam[0] > lam[1]                                # more goals expected for S
