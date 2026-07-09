"""Decompose the market's 2026 edge: information flow or model quality? (PAPER.md §5.2)

Three diagnostics on the played 2026 matches:
  1. log-loss by segment (matchday 1/2/3, knockout) — if the market's edge grows with
     the tournament, it is information (lineups, rotations, injuries), not modeling;
  2. an in-tournament-updating Dixon-Coles (refit before every match date) — tests
     whether score-updating closes the gap (it does not; the missing info is news);
  3. a minnow split (any team ranked >60 at the freeze) — where are models vs market
     stronger?

    python scripts/market_edge_decomposition.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from groll_eval import (                                                    # noqa: E402
    TRAIN_YEARS,
    build_frame,
    build_snapshots,
    fit_abilities,
    load_market_probs,
    wc_matches,
    wc_start,
)
from wc_trader.backtest.metrics import paired_logloss_bootstrap            # noqa: E402
from wc_trader.backtest.simulator import CLASSES, match_outcome            # noqa: E402
from wc_trader.data.fifa_rank import RankLookup, load_fifa_rankings        # noqa: E402
from wc_trader.data.lineups import fetch_lineups, load_lineups             # noqa: E402
from wc_trader.data.results import load_results                            # noqa: E402
from wc_trader.model.dixon_coles import DixonColesModel                    # noqa: E402
from wc_trader.model.elo import EloModel                                   # noqa: E402
from wc_trader.model.lineup_adjust import (                                # noqa: E402
    NewsAdjustedDC,
    NewsCoefficients,
    NewsFeatureBuilder,
)
from wc_trader.types import Outcome                                        # noqa: E402


def main() -> None:
    df = load_results()
    wc = (df[(df.tournament == "FIFA World Cup") & (df.date.dt.year == 2026)]
          .sort_values("date").reset_index(drop=True))
    market = load_market_probs("data/raw/wc2026_odds.csv")
    freeze = pd.Timestamp(wc_start(2026))

    dc_frozen = fit_abilities(df, 2026)
    elo = EloModel()
    for r in df[df.date < freeze].itertuples(index=False):
        elo.update(r.home_team, r.away_team, r.home_score, r.away_score, bool(r.neutral))

    dc_by_date = {}
    for d in sorted(wc.date.unique()):
        d = pd.Timestamp(d)
        train = df[(df.date >= d - pd.DateOffset(years=8)) & (df.date < d)]
        dc_by_date[d] = DixonColesModel().fit(train, half_life_days=540.0, min_matches=8)

    # news layer: frozen models + official lineups, coefficients calibrated on 2018+2022
    import json as _json
    from pathlib import Path as _Path
    coef_all = _json.loads(_Path("paper/news_coefficients.json").read_text())
    coef_dc = NewsCoefficients(**{k: coef_all["dc"][k] for k in ("beta_own", "beta_opp", "gamma")})
    coef_rf = NewsCoefficients(**{k: coef_all["rf"][k] for k in ("beta_own", "beta_opp", "gamma")})
    fetch_lineups(2026)
    recs26 = load_lineups(2026)
    builder = NewsFeatureBuilder(dc_frozen, recs26)
    dc_news = NewsAdjustedDC(dc_frozen, builder, coef_dc, use_form=True)
    dc_lineup_only = NewsAdjustedDC(dc_frozen, builder, coef_dc, use_form=False)

    # hybrid RF (trained 1998-2022, frozen) wrapped with the same machinery
    from paper_eval import rf_lambda_table
    from wc_trader.data.fifa_rank import fetch_fifa_rankings, fetch_prewc2026_snapshot
    from wc_trader.data.worldbank import IndicatorLookup, fetch_indicator
    from wc_trader.data.wc_meta import iso3
    from wc_trader.model.groll_rf import HybridRF
    from wc_trader.model.lineup_adjust import LambdaGridAdapter

    fetch_fifa_rankings()
    fetch_prewc2026_snapshot()
    ranks = RankLookup(load_fifa_rankings())
    universe = sorted({t for y in TRAIN_YEARS + [2026] for m in [wc_matches(df, y)]
                       for t in set(m.home_team) | set(m.away_team)})
    codes = sorted({c for t in universe if (c := iso3(t))})
    gdp = IndicatorLookup(fetch_indicator(codes, "gdp_pc"))
    pop = IndicatorLookup(fetch_indicator(codes, "population"))
    missing: dict = {}
    frames = {y: build_frame(df, y, ranks, gdp, pop, missing)[0] for y in TRAIN_YEARS}
    rf26 = HybridRF().fit(pd.concat(frames.values(), ignore_index=True))
    teams26 = sorted(set(wc.home_team) | set(wc.away_team))
    snaps26 = build_snapshots(teams26, 2026, dc_frozen, ranks, gdp, pop, missing)
    rf_adapter = LambdaGridAdapter(rf_lambda_table(rf26, snaps26))
    rf_builder = NewsFeatureBuilder(rf_adapter, recs26)
    rf_news = NewsAdjustedDC(rf_adapter, rf_builder, coef_rf, use_form=True)
    rf_adapter_probs = NewsAdjustedDC(rf_adapter, rf_builder,
                                      NewsCoefficients(0.0, 0.0, 0.0))  # frozen RF

    rank_table = RankLookup(load_fifa_rankings()).table_asof(freeze)
    rank_of = lambda t: rank_table.get(t, (None, 999))[1]

    rows = []
    for i, r in enumerate(wc.itertuples(index=False)):
        neutral = bool(r.neutral)
        def probs_of(model):
            if isinstance(model, NewsAdjustedDC):   # date disambiguates repeat pairings
                p = model.match_probabilities(r.home_team, r.away_team, neutral,
                                              date=str(r.date.date()))
            else:
                p = model.match_probabilities(r.home_team, r.away_team, neutral)
            return {c: p[Outcome(c)] for c in CLASSES}
        rows.append({
            "y": match_outcome(int(r.home_score), int(r.away_score)),
            "minnow": max(rank_of(r.home_team), rank_of(r.away_team)) > 60,
            "probs": {
                "Market": market[(r.home_team, r.away_team, str(r.date.date()))],
                "DC frozen": probs_of(dc_frozen),
                "DC updated": probs_of(dc_by_date[pd.Timestamp(r.date)]),
                "DC+lineup": probs_of(dc_lineup_only),
                "DC+lineup+form": probs_of(dc_news),
                "RF frozen": probs_of(rf_adapter_probs),
                "RF+news": probs_of(rf_news),
                "Elo updated": probs_of(elo),
            },
        })
        elo.update(r.home_team, r.away_team, r.home_score, r.away_score, neutral)

    def ll(subset, model):
        return sum(-math.log(max(1e-15, x["probs"][model][x["y"]])) for x in subset) / len(subset)

    models = ["Market", "DC frozen", "DC updated", "DC+lineup", "DC+lineup+form",
              "RF frozen", "RF+news", "Elo updated"]
    segs = {"MD1 (1-24)": rows[:24], "MD2 (25-48)": rows[24:48],
            "MD3 (49-72)": rows[48:72], "KO (73-96)": rows[72:96], "ALL": rows}
    print(f"{'segment':<14}" + "".join(f"{m:>13}" for m in models))
    for name, s in segs.items():
        print(f"{name:<14}" + "".join(f"{ll(s, m):>13.4f}" for m in models))

    print("\nminnow split (any team ranked >60 at freeze):")
    for flag, label in ((True, "minnow matches"), (False, "established only")):
        s = [x for x in rows if x["minnow"] == flag]
        print(f"  {label:<18} n={len(s):>3}  "
              + "  ".join(f"{m} {ll(s, m):.4f}" for m in models))

    ys = [x["y"] for x in rows]
    for a, b in (("DC updated", "Market"), ("DC+lineup+form", "Market"),
                 ("DC+lineup+form", "DC frozen"), ("RF+news", "RF frozen"),
                 ("RF+news", "Market")):
        bs = paired_logloss_bootstrap([x["probs"][a] for x in rows],
                                      [x["probs"][b] for x in rows], ys)
        print(f"\n{a} − {b}: {bs['mean_diff']:+.4f} [{bs['ci_low']:+.4f}, {bs['ci_high']:+.4f}]")


if __name__ == "__main__":
    main()
