"""The Groll et al. (2019) experiment: train on World Cups 1998–2022, predict the 2026
World Cup out-of-sample, and compare against Dixon-Coles, Elo, and base rate.

    python groll_eval.py                # full report (downloads/caches data on first run)
    python groll_eval.py --no-refresh   # skip re-downloading the results dataset

Protocol (GROLL_PLAN.md §3): every feature is frozen at tournament start — abilities are
fit on matches strictly before it, FIFA rank is the last release ≤ it, GDP/population the
latest year before it. Nothing post-kickoff enters any model.
"""
from __future__ import annotations

import argparse
import math

import pandas as pd

from wc_trader.backtest.metrics import (
    accuracy,
    brier_score,
    log_loss,
    paired_logloss_bootstrap,
)
from wc_trader.backtest.simulator import CLASSES, match_outcome
from wc_trader.data.fifa_rank import (
    RankLookup,
    fetch_fifa_rankings,
    fetch_prewc2026_snapshot,
    load_fifa_rankings,
)
from wc_trader.data.results import fetch_results, load_results
from wc_trader.data.wc_meta import WORLD_CUPS, confederation, is_host, iso3, wc_start
from wc_trader.data.worldbank import IndicatorLookup, fetch_indicator
from wc_trader.model.dixon_coles import DixonColesModel
from wc_trader.model.elo import EloModel
from wc_trader.model.groll_rf import HybridRF, TeamSnapshot, match_rows
from wc_trader.types import Outcome

TRAIN_YEARS = [1998, 2002, 2006, 2010, 2014, 2018, 2022]
TEST_YEAR = 2026
ABILITY_WINDOW_YEARS = 8
ABILITY_HALF_LIFE = 540.0
GROUP_STAGE_MATCHES = {2026: 72}  # 48-team format: first 72 matches are group stage
ODDS_CSV = "data/raw/wc2026_odds.csv"  # scripts/scrape_wc2026_odds.py output (optional)


def load_market_probs(path: str = ODDS_CSV) -> dict[tuple, dict]:
    """{(home, away, date): de-vigged {HOME/DRAW/AWAY: prob}} from scraped odds, or {}."""
    from pathlib import Path
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


def wc_matches(df: pd.DataFrame, year: int) -> pd.DataFrame:
    m = df[(df.tournament == "FIFA World Cup") & (df.date.dt.year == year)]
    return m.sort_values("date").reset_index(drop=True)


def fit_abilities(df: pd.DataFrame, year: int) -> DixonColesModel:
    start = pd.Timestamp(wc_start(year))
    lo = start - pd.DateOffset(years=ABILITY_WINDOW_YEARS)
    train = df[(df.date >= lo) & (df.date < start)]
    return DixonColesModel().fit(train, half_life_days=ABILITY_HALF_LIFE, min_matches=8)


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


def build_frame(df: pd.DataFrame, year: int, ranks: RankLookup,
                gdp: IndicatorLookup, pop: IndicatorLookup, missing: dict,
                dc: DixonColesModel | None = None) -> tuple[pd.DataFrame, DixonColesModel]:
    dc = dc or fit_abilities(df, year)
    matches = wc_matches(df, year)
    teams = sorted(set(matches.home_team) | set(matches.away_team))
    snaps = build_snapshots(teams, year, dc, ranks, gdp, pop, missing)
    rows = []
    for i, r in enumerate(matches.itertuples(index=False)):
        rows.extend(match_rows(f"{year}-{i:03d}", r.home_team, r.away_team,
                               int(r.home_score), int(r.away_score), snaps))
    return pd.DataFrame(rows), dc


def report_block(title: str, probs_by_model: dict[str, list[dict]], outcomes: list[str]) -> None:
    print(f"\n  {title}  (n={len(outcomes)})")
    print(f"    {'model':<14}{'log-loss':>10}{'Brier':>9}{'accuracy':>10}")
    for name, probs in probs_by_model.items():
        print(f"    {name:<14}{log_loss(probs, outcomes):>10.4f}"
              f"{brier_score(probs, outcomes, CLASSES):>9.4f}"
              f"{accuracy(probs, outcomes):>9.1%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-refresh", action="store_true")
    ap.add_argument("--trees", type=int, default=1000)
    args = ap.parse_args()

    if not args.no_refresh:
        fetch_results(force=True)
    df = load_results()
    fetch_fifa_rankings()
    try:
        fetch_prewc2026_snapshot()   # official 2026-06-11 release via FIFA's live API
    except Exception as e:
        print(f"warning: 2026 ranking snapshot unavailable ({e}); falling back to last cached release")
    ranks = RankLookup(load_fifa_rankings())

    all_years = TRAIN_YEARS + [TEST_YEAR]
    universe = sorted({t for y in all_years for m in [wc_matches(df, y)]
                       for t in set(m.home_team) | set(m.away_team)})
    codes = sorted({c for t in universe if (c := iso3(t))})
    gdp = IndicatorLookup(fetch_indicator(codes, "gdp_pc"))
    pop = IndicatorLookup(fetch_indicator(codes, "population"))

    print(f"Team universe: {len(universe)} teams across {len(all_years)} World Cups")
    missing: dict = {}

    # --- training frame (1998–2022) ---
    frames = []
    for y in TRAIN_YEARS:
        frame, _ = build_frame(df, y, ranks, gdp, pop, missing)
        frames.append(frame)
        print(f"  WC {y}: {len(frame)//2} matches -> {len(frame)} rows")
    train = pd.concat(frames, ignore_index=True)

    rf = HybridRF(n_estimators=args.trees).fit(train)
    print(f"\nHybrid RF trained on {len(train)} rows ({len(train)//2} matches).")

    print("\nVariable importance (paper's claim: abilities dominate):")
    for name, v in rf.importances().head(10).items():
        print(f"    {name:<18}{v:.3f}")

    # --- 2026 out-of-sample evaluation ---
    test_matches = wc_matches(df, TEST_YEAR)
    if test_matches.empty:
        print("\nNo 2026 matches in the dataset yet.")
        return
    stale = ranks.staleness_days(wc_start(TEST_YEAR))
    print(f"\n2026 evaluation: {len(test_matches)} matches played. "
          f"FIFA-rank staleness at freeze: {stale} days.")

    test_frame, dc26 = build_frame(df, TEST_YEAR, ranks, gdp, pop, missing)

    elo = EloModel()
    freeze = pd.Timestamp(wc_start(TEST_YEAR))
    for r in df[df.date < freeze].itertuples(index=False):
        elo.update(r.home_team, r.away_team, r.home_score, r.away_score, bool(r.neutral))

    train_matches = pd.concat([wc_matches(df, y) for y in TRAIN_YEARS])
    base_counts = train_matches.apply(
        lambda r: match_outcome(int(r.home_score), int(r.away_score)), axis=1).value_counts()
    base = {c: float(base_counts.get(c, 0)) / len(train_matches) for c in CLASSES}

    market = load_market_probs()
    probs: dict[str, list[dict]] = {"Hybrid RF": [], "Dixon-Coles": [], "Elo": [], "Base rate": []}
    market_probs: list[dict | None] = []
    outcomes: list[str] = []
    for i, r in enumerate(test_matches.itertuples(index=False)):
        pair = test_frame.iloc[2 * i: 2 * i + 2]
        assert pair.iloc[0]["team"] == r.home_team  # row order sanity
        rf_p = rf.match_probabilities_from_rows(pair)
        dc_p = dc26.match_probabilities(r.home_team, r.away_team, bool(r.neutral))
        el_p = elo.match_probabilities(r.home_team, r.away_team, bool(r.neutral))
        probs["Hybrid RF"].append({c: rf_p[Outcome(c)] for c in CLASSES})
        probs["Dixon-Coles"].append({c: dc_p[Outcome(c)] for c in CLASSES})
        probs["Elo"].append({c: el_p[Outcome(c)] for c in CLASSES})
        probs["Base rate"].append(base)
        market_probs.append(market.get((r.home_team, r.away_team, str(r.date.date()))))
        outcomes.append(match_outcome(int(r.home_score), int(r.away_score)))

    if market and all(m is not None for m in market_probs):
        probs["Market (odds)"] = market_probs  # full coverage -> same eval set
    elif market:
        n_cov = sum(m is not None for m in market_probs)
        print(f"warning: odds cover only {n_cov}/{len(market_probs)} matches — market row skipped")

    print(f"\n{'=' * 62}\nGroll et al. (2019) hybrid RF — 2026 World Cup, out-of-sample\n{'=' * 62}")
    report_block("All matches", probs, outcomes)

    def paired(a: str, b: str) -> None:
        bs = paired_logloss_bootstrap(probs[a], probs[b], outcomes)
        sig = "significant" if (bs["ci_low"] > 0 or bs["ci_high"] < 0) else "NOT significant at 95%"
        print(f"  Paired {a}−{b} log-loss diff: {bs['mean_diff']:+.4f} "
              f"[{bs['ci_low']:+.4f}, {bs['ci_high']:+.4f}]  ({sig}, n={bs['n']})")

    print()
    paired("Hybrid RF", "Dixon-Coles")
    if "Market (odds)" in probs:
        paired("Hybrid RF", "Market (odds)")
        paired("Dixon-Coles", "Market (odds)")

    n_group = min(GROUP_STAGE_MATCHES.get(TEST_YEAR, len(outcomes)), len(outcomes))
    if n_group < len(outcomes):
        report_block("Group stage only (90-min results; no ET distortion)",
                     {k: v[:n_group] for k, v in probs.items()}, outcomes[:n_group])
        report_block("Knockout so far (scores incl. extra time)",
                     {k: v[n_group:] for k, v in probs.items()}, outcomes[n_group:])

    if missing:
        print("\nCovariate gaps (median-filled in RF):")
        for k, items in missing.items():
            print(f"    {k:<8} {len(items):>3} team-tournaments  e.g. {items[:4]}")
    print(f"\nUniform-guess log-loss for reference: {math.log(3):.4f}")


if __name__ == "__main__":
    main()
