"""Shared machinery for the replication experiments (used by paper_eval and scripts/).

Everything here enforces the freeze protocol: for a tournament, abilities are fit on
matches strictly before its start, FIFA rank is the last release at or before it, and
World Bank covariates the latest year strictly before it.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data.fifa_rank import RankLookup, fetch_fifa_rankings, fetch_prewc2026_snapshot, load_fifa_rankings
from .data.wc_meta import confederation, is_host, iso3, wc_start
from .data.worldbank import IndicatorLookup, fetch_indicator
from .model.dixon_coles import DixonColesModel
from .model.elo import EloModel
from .model.groll_rf import TeamSnapshot, match_rows

TRAIN_YEARS = [1998, 2002, 2006, 2010, 2014, 2018, 2022]
TEST_YEAR = 2026
ABILITY_WINDOW_YEARS = 8
ABILITY_HALF_LIFE = 540.0


def wc_matches(df: pd.DataFrame, year: int) -> pd.DataFrame:
    m = df[(df.tournament == "FIFA World Cup") & (df.date.dt.year == year)]
    return m.sort_values("date").reset_index(drop=True)


def fit_abilities(df: pd.DataFrame, year: int) -> DixonColesModel:
    start = pd.Timestamp(wc_start(year))
    lo = start - pd.DateOffset(years=ABILITY_WINDOW_YEARS)
    train = df[(df.date >= lo) & (df.date < start)]
    return DixonColesModel().fit(train, half_life_days=ABILITY_HALF_LIFE, min_matches=8)


def frozen_elo(df: pd.DataFrame, year: int) -> EloModel:
    elo = EloModel()
    freeze = pd.Timestamp(wc_start(year))
    for r in df[df.date < freeze].itertuples(index=False):
        elo.update(r.home_team, r.away_team, r.home_score, r.away_score, bool(r.neutral))
    return elo


def covariate_lookups(df: pd.DataFrame, years: list[int] | None = None):
    """(ranks, gdp, pop) lookups covering every team in the given World Cups."""
    years = years or (TRAIN_YEARS + [TEST_YEAR])
    fetch_fifa_rankings()
    try:
        fetch_prewc2026_snapshot()
    except Exception as e:  # pragma: no cover - network path
        print(f"warning: 2026 ranking snapshot unavailable ({e}); using last cached release")
    ranks = RankLookup(load_fifa_rankings())
    universe = sorted({t for y in years for m in [wc_matches(df, y)]
                       for t in set(m.home_team) | set(m.away_team)})
    codes = sorted({c for t in universe if (c := iso3(t))})
    gdp = IndicatorLookup(fetch_indicator(codes, "gdp_pc"))
    pop = IndicatorLookup(fetch_indicator(codes, "population"))
    return ranks, gdp, pop


def build_snapshots(teams: list[str], year: int, dc: DixonColesModel,
                    ranks: RankLookup, gdp: IndicatorLookup, pop: IndicatorLookup,
                    missing: dict) -> dict[str, TeamSnapshot]:
    start = pd.Timestamp(wc_start(year))
    rank_table = ranks.table_asof(start)
    snaps = {}
    for t in teams:
        if t not in dc.attack:
            missing.setdefault("ability", []).append((year, t))
        rp = rank_table.get(t)
        if rp is None:
            missing.setdefault("rank", []).append((year, t))
        code = iso3(t)
        g = gdp.value_asof(code, start.year - 1)
        p = pop.value_asof(code, start.year - 1)
        if g is None:
            missing.setdefault("gdp", []).append((year, t))
        if confederation(t) == "OTHER":
            missing.setdefault("confed", []).append((year, t))
        snaps[t] = TeamSnapshot(
            attack=dc.attack.get(t, 0.0), defense=dc.defense.get(t, 0.0),
            rank_points=rp[0] if rp else None, rank_pos=rp[1] if rp else None,
            gdp_pc=g, population=p, host=is_host(t, year), confed=confederation(t),
        )
    return snaps


def build_frame(df: pd.DataFrame, year: int, ranks, gdp, pop, missing: dict,
                dc: DixonColesModel | None = None) -> tuple[pd.DataFrame, DixonColesModel]:
    """Two-rows-per-match training/eval frame for one tournament (+ its ability model)."""
    dc = dc or fit_abilities(df, year)
    matches = wc_matches(df, year)
    teams = sorted(set(matches.home_team) | set(matches.away_team))
    snaps = build_snapshots(teams, year, dc, ranks, gdp, pop, missing)
    rows = []
    for i, r in enumerate(matches.itertuples(index=False)):
        rows.extend(match_rows(f"{year}-{i:03d}", r.home_team, r.away_team,
                               int(r.home_score), int(r.away_score), snaps))
    return pd.DataFrame(rows), dc


def load_market_probs(path: str = "data/raw/wc2026_odds.csv") -> dict[tuple, dict]:
    """{(home, away, date): de-vigged {HOME/DRAW/AWAY: prob}} from scraped odds, or {}."""
    if not Path(path).exists():
        return {}
    odds = pd.read_csv(path)
    out = {}
    for r in odds.itertuples(index=False):
        inv = [1 / r.odds_home, 1 / r.odds_draw, 1 / r.odds_away]
        tot = sum(inv)
        out[(r.home_team, r.away_team, str(r.date))] = {
            "HOME": inv[0] / tot, "DRAW": inv[1] / tot, "AWAY": inv[2] / tot}
    return out
