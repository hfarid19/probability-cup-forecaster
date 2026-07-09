"""Who wins the 2026 World Cup? Exact bracket forecast from the frozen models.

Uses the same freeze-at-tournament-start protocol as groll_eval.py (abilities, rankings,
Elo all fit on pre-tournament data only), conditioned on the actual remaining bracket:

    QF97 France v Morocco ─┐
    QF98 Spain  v Belgium ─┴─ SF1 (Jul 14)
    QF99 Norway v England ─┐
    QF100 Argentina v Switzerland ─┴─ SF2 (Jul 15)   → Final Jul 19

All hosts are eliminated, so every remaining match is neutral-venue.

    python champion_forecast.py
"""
from __future__ import annotations

import pandas as pd

from groll_eval import TRAIN_YEARS, build_frame, build_snapshots, fit_abilities, wc_start
from wc_trader.backtest.bracket import advance_prob_from_3way, solve_bracket
from wc_trader.data.fifa_rank import (
    RankLookup,
    fetch_fifa_rankings,
    fetch_prewc2026_snapshot,
    load_fifa_rankings,
)
from wc_trader.data.results import fetch_results, load_results
from wc_trader.data.wc_meta import iso3
from wc_trader.data.worldbank import IndicatorLookup, fetch_indicator
from wc_trader.model.elo import EloModel
from wc_trader.model.groll_rf import HybridRF, match_rows
from wc_trader.types import Outcome

# Actual bracket after the Round of 16 (FIFA match numbers 97-100; SF1 = W97 v W98).
QF_PAIRS = [
    ("France", "Morocco"),
    ("Spain", "Belgium"),
    ("Norway", "England"),
    ("Argentina", "Switzerland"),
]
ALIVE = [t for pair in QF_PAIRS for t in pair]


def main() -> None:
    fetch_results()
    df = load_results()
    fetch_fifa_rankings()
    fetch_prewc2026_snapshot()
    ranks = RankLookup(load_fifa_rankings())

    all_years = TRAIN_YEARS + [2026]
    universe = sorted({t for y in all_years
                       for m in [df[(df.tournament == "FIFA World Cup") & (df.date.dt.year == y)]]
                       for t in set(m.home_team) | set(m.away_team)})
    codes = sorted({c for t in universe if (c := iso3(t))})
    gdp = IndicatorLookup(fetch_indicator(codes, "gdp_pc"))
    pop = IndicatorLookup(fetch_indicator(codes, "population"))

    # Frozen models (identical to groll_eval protocol)
    missing: dict = {}
    frames = [build_frame(df, y, ranks, gdp, pop, missing)[0] for y in TRAIN_YEARS]
    rf = HybridRF().fit(pd.concat(frames, ignore_index=True))
    dc = fit_abilities(df, 2026)
    snaps = build_snapshots(ALIVE, 2026, dc, ranks, gdp, pop, missing)

    elo = EloModel()
    freeze = pd.Timestamp(wc_start(2026))
    for r in df[df.date < freeze].itertuples(index=False):
        elo.update(r.home_team, r.away_team, r.home_score, r.away_score, bool(r.neutral))

    def p_beat_dc(a: str, b: str) -> float:
        p = dc.match_probabilities(a, b, neutral=True)
        return advance_prob_from_3way(p[Outcome.HOME], p[Outcome.DRAW])

    def p_beat_elo(a: str, b: str) -> float:
        p = elo.match_probabilities(a, b, neutral=True)
        return advance_prob_from_3way(p[Outcome.HOME], p[Outcome.DRAW])

    def p_beat_rf(a: str, b: str) -> float:
        pair = pd.DataFrame(match_rows("fc", a, b, None, None, snaps))
        p = rf.match_probabilities_from_rows(pair)
        return advance_prob_from_3way(p[Outcome.HOME], p[Outcome.DRAW])

    forecasts = {
        "Hybrid RF": solve_bracket(QF_PAIRS, p_beat_rf),
        "Dixon-Coles": solve_bracket(QF_PAIRS, p_beat_dc),
        "Elo": solve_bracket(QF_PAIRS, p_beat_elo),
    }

    print(f"\n{'=' * 66}\n2026 World Cup champion forecast — models frozen at 2026-06-11\n{'=' * 66}")
    print("Bracket: (France-Morocco / Spain-Belgium) -> SF1;"
          " (Norway-England / Argentina-Switzerland) -> SF2\n")
    order = sorted(ALIVE, key=lambda t: -forecasts["Dixon-Coles"].p_champion[t])
    header = f"{'team':<14}" + "".join(
        f"{name + ' win%':>16}" for name in forecasts)
    print("  P(champion):")
    print(f"    {'team':<13}" + "".join(f"{n:>14}" for n in forecasts))
    for t in order:
        print(f"    {t:<13}" + "".join(f"{fc.p_champion[t]:>13.1%} " for fc in forecasts.values()))

    print("\n  P(win quarterfinal) / P(reach final)  [Dixon-Coles]:")
    dcf = forecasts["Dixon-Coles"]
    for t in order:
        print(f"    {t:<13} QF {dcf.p_qf[t]:>6.1%}   final {dcf.p_final[t]:>6.1%}")

    for name, fc in forecasts.items():
        s = sum(fc.p_champion.values())
        assert abs(s - 1.0) < 1e-9, (name, s)


if __name__ == "__main__":
    main()
