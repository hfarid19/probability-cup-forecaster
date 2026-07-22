"""Produce every number for PAPER.md (the Groll et al. replication paper).

Sections map one-to-one to the paper:
  §4  Leave-one-tournament-out comparison over WCs 1998 to 2022:
      hybrid RF vs Dixon-Coles vs Elo on classification rate, likelihood, RPS (+ log-loss).
  §5a The 2022 World Cup: train ≤2018, evaluate all 64 matches (incl. market benchmark),
      variable importance, pre-tournament Monte-Carlo simulation (champion/final/SF probs).
  §5b The 2026 World Cup (through the Round of 16): same, plus the conditional
      quarterfinal-bracket forecast and the market's outright odds.

Everything lands in paper/results.json.

    python paper_eval.py            # ~5 minutes; add --sims 5000 for a quick pass
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from wc_trader.backtest.bracket import advance_prob_from_3way, solve_bracket
from wc_trader.backtest.metrics import (
    CLASSES,
    accuracy,
    avg_likelihood,
    log_loss,
    match_outcome,
    paired_logloss_bootstrap,
    rps,
)
from wc_trader.backtest.tournament import dc_sampler, parse_calendar, rf_sampler, simulate
from wc_trader.data.results import fetch_results, load_results
from wc_trader.experiment import (
    TRAIN_YEARS,
    build_frame,
    build_snapshots,
    covariate_lookups,
    frozen_elo,
    load_market_probs,
    wc_matches,
)
from wc_trader.model.groll_rf import HybridRF, match_rows, rf_lambda_table
from wc_trader.types import Outcome

QF_PAIRS_2026 = [("France", "Morocco"), ("Spain", "Belgium"),
                 ("Norway", "England"), ("Argentina", "Switzerland")]


def metric_block(probs: list[dict], outcomes: list[str]) -> dict:
    """Compute the paper's headline scoring metrics for one set of forecasts.

    Args:
        probs: Per-match forecasts, each a dict over the outcome classes.
        outcomes: Realized outcome class for each match.

    Returns:
        dict: {"class_rate", "likelihood", "rps", "log_loss"} for the forecasts.
    """
    return {
        "class_rate": accuracy(probs, outcomes),
        "likelihood": avg_likelihood(probs, outcomes),
        "rps": rps(probs, outcomes),
        "log_loss": log_loss(probs, outcomes),
    }


def eval_tournament(df, year, frame, dc, elo, rf, market: dict | None) -> dict:
    """Produce per-match forecasts for every model on one tournament.

    Scores each match with the hybrid RF, Dixon-Coles, and Elo models, and includes
    the market forecasts too when de-vigged odds are available for every match.

    Args:
        df: The full results dataset.
        year: The World Cup year to evaluate.
        frame: The two-rows-per-match feature frame for the tournament.
        dc: The fitted Dixon-Coles model.
        elo: The frozen Elo model.
        rf: The fitted hybrid RF model.
        market: De-vigged market probabilities, or None to skip the market model.

    Returns:
        dict: {"probs": {model: [per-match forecast dicts]}, "outcomes": [classes]}.
    """
    matches = wc_matches(df, year)
    probs = {"Hybrid RF": [], "Dixon-Coles": [], "Elo": []}
    mkt: list[dict] = []
    outcomes = []
    for i, r in enumerate(matches.itertuples(index=False)):
        pair = frame.iloc[2 * i: 2 * i + 2]
        assert pair.iloc[0]["team"] == r.home_team
        for name, p in (("Hybrid RF", rf.match_probabilities_from_rows(pair)),
                        ("Dixon-Coles", dc.match_probabilities(r.home_team, r.away_team, bool(r.neutral))),
                        ("Elo", elo.match_probabilities(r.home_team, r.away_team, bool(r.neutral)))):
            probs[name].append({c: p[Outcome(c)] for c in CLASSES})
        outcomes.append(match_outcome(int(r.home_score), int(r.away_score)))
        if market is not None:
            m = market.get((r.home_team, r.away_team, str(r.date.date())))
            if m:
                mkt.append(m)
    if market is not None and len(mkt) == len(outcomes):
        probs["Market"] = mkt
    return {"probs": probs, "outcomes": outcomes}


def sim_tables(spec, dc, rf, snaps, n_sims: int) -> dict:
    """Run the Monte-Carlo simulator with both samplers and summarize the results.

    Simulates the tournament under the Dixon-Coles and hybrid RF score samplers, then
    keeps the top champions and most likely finals for each.

    Args:
        spec: The parsed tournament template.
        dc: The fitted Dixon-Coles model (Dixon-Coles sampler).
        rf: The fitted hybrid RF model (RF sampler).
        snaps: Per-team feature snapshots feeding the RF expected-goals table.
        n_sims: Number of tournament simulations per sampler.

    Returns:
        dict: Top champion tables and top finals pairings for both models.
    """
    dc_res, dc_finals = simulate(spec, dc_sampler(dc), n_sims=n_sims, seed=42)
    rf_res, rf_finals = simulate(spec, rf_sampler(rf_lambda_table(rf, snaps)),
                                 n_sims=n_sims, seed=42)
    def top(res, k=12):
        """Keep the k teams with the highest champion probability.

        Args:
            res: Team to its stage-probability dict.
            k: Number of teams to keep.

        Returns:
            dict: The k highest-champion-probability teams and their probabilities.
        """
        return {t: r for t, r in sorted(res.items(), key=lambda x: -x[1]["champion"])[:k]}
    def toppairs(fp, k=5):
        """Format the k most likely finals pairings as "A v B" keys.

        Args:
            fp: Finalist pair tuple to its probability, most likely first.
            k: Number of pairings to keep.

        Returns:
            dict: "teamA v teamB" to probability for the top k pairings.
        """
        return {" v ".join(p): v for p, v in list(fp.items())[:k]}
    return {"Dixon-Coles": top(dc_res), "Hybrid RF": top(rf_res),
            "finals_dc": toppairs(dc_finals), "finals_rf": toppairs(rf_finals)}


def main() -> None:
    """Run the full replication analysis and write paper/results.json.

    Loads the results data and frozen covariates, builds per-tournament frames, then
    produces every number for the paper: the leave-one-tournament-out comparison
    (section 4), the 2022 experiment (section 5a), and the 2026 experiment through
    the Round of 16 (section 5b). Command-line flags control the simulation count and
    whether to refresh the results dataset.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--no-refresh", action="store_true")
    args = ap.parse_args()

    if not args.no_refresh:
        fetch_results(force=True)
    df = load_results()
    ranks, gdp, pop = covariate_lookups(df)
    missing: dict = {}

    print("Building per-tournament frames (freeze-date covariates)...")
    frames: dict[int, pd.DataFrame] = {}
    dcs: dict[int, object] = {}
    for y in TRAIN_YEARS + [2026]:
        frames[y], dcs[y] = build_frame(df, y, ranks, gdp, pop, missing)
        print(f"  WC {y}: {len(frames[y]) // 2} matches")

    results: dict = {"meta": {"n_sims": args.sims,
                              "train_years": TRAIN_YEARS,
                              "note": "all covariates frozen at each tournament's start"}}

    # ---------- §4 leave-one-tournament-out ----------
    print("\n§4 Leave-one-tournament-out comparison...")
    loo: dict = {}
    pooled = {m: {"probs": [], "outcomes": []} for m in ("Hybrid RF", "Dixon-Coles", "Elo")}
    for y in TRAIN_YEARS:
        train = pd.concat([frames[k] for k in TRAIN_YEARS if k != y], ignore_index=True)
        rf = HybridRF().fit(train)
        ev = eval_tournament(df, y, frames[y], dcs[y], frozen_elo(df, y), rf, market=None)
        loo[y] = {m: metric_block(p, ev["outcomes"]) for m, p in ev["probs"].items()}
        for m in pooled:
            pooled[m]["probs"].extend(ev["probs"][m])
            pooled[m]["outcomes"] = pooled[m]["outcomes"] + ev["outcomes"]
        print(f"  held out {y}: RF rps {loo[y]['Hybrid RF']['rps']:.4f} "
              f"| DC {loo[y]['Dixon-Coles']['rps']:.4f} | Elo {loo[y]['Elo']['rps']:.4f}")
    results["loo"] = {"per_year": loo,
                      "pooled": {m: metric_block(v["probs"], v["outcomes"])
                                 for m, v in pooled.items()}}

    # ---------- §5a the 2022 World Cup ----------
    print("\n§5a 2022 experiment (train ≤2018)...")
    train22 = pd.concat([frames[y] for y in TRAIN_YEARS if y < 2022], ignore_index=True)
    rf22 = HybridRF().fit(train22)
    market22 = load_market_probs("data/raw/wc2022_odds.csv")
    ev22 = eval_tournament(df, 2022, frames[2022], dcs[2022], frozen_elo(df, 2022),
                           rf22, market22)
    wc22: dict = {"eval": {m: metric_block(p, ev22["outcomes"]) for m, p in ev22["probs"].items()},
                  "importance": rf22.importances().head(10).round(4).to_dict(),
                  "bootstrap": {}}
    for a, b in (("Hybrid RF", "Dixon-Coles"), ("Hybrid RF", "Market"), ("Dixon-Coles", "Market")):
        if b in ev22["probs"]:
            wc22["bootstrap"][f"{a} - {b}"] = paired_logloss_bootstrap(
                ev22["probs"][a], ev22["probs"][b], ev22["outcomes"])
    teams22 = sorted(set(wc_matches(df, 2022).home_team) | set(wc_matches(df, 2022).away_team))
    snaps22 = build_snapshots(teams22, 2022, dcs[2022], ranks, gdp, pop, missing)
    spec22 = parse_calendar("data/raw/fifa_calendar_2022.json")
    print("  simulating 2022 tournament...")
    wc22["sim"] = sim_tables(spec22, dcs[2022], rf22, snaps22, args.sims)
    wc22["actual"] = {"champion": "Argentina", "runner_up": "France",
                      "note": "final drawn 3-3 after ET, Argentina won on penalties"}
    results["wc2022"] = wc22

    # ---------- §5b the 2026 World Cup (to date) ----------
    print("\n§5b 2026 experiment (train ≤2022, matches through Round of 16)...")
    train26 = pd.concat([frames[y] for y in TRAIN_YEARS], ignore_index=True)
    rf26 = HybridRF().fit(train26)
    market26 = load_market_probs("data/raw/wc2026_odds.csv")
    ev26 = eval_tournament(df, 2026, frames[2026], dcs[2026], frozen_elo(df, 2026),
                           rf26, market26)
    wc26: dict = {"eval": {m: metric_block(p, ev26["outcomes"]) for m, p in ev26["probs"].items()},
                  "n_matches": len(ev26["outcomes"]),
                  "importance": rf26.importances().head(10).round(4).to_dict(),
                  "bootstrap": {}}
    for a, b in (("Hybrid RF", "Dixon-Coles"), ("Hybrid RF", "Market"), ("Dixon-Coles", "Market")):
        if b in ev26["probs"]:
            wc26["bootstrap"][f"{a} - {b}"] = paired_logloss_bootstrap(
                ev26["probs"][a], ev26["probs"][b], ev26["outcomes"])
    spec26 = parse_calendar("data/raw/fifa_calendar_2026.json")
    snaps26 = build_snapshots(spec26.teams, 2026, dcs[2026], ranks, gdp, pop, missing)
    print("  simulating 2026 tournament (pre-tournament view)...")
    wc26["sim"] = sim_tables(spec26, dcs[2026], rf26, snaps26, args.sims)

    # conditional forecast given the actual quarterfinal bracket
    elo26 = frozen_elo(df, 2026)

    def beat(model_probs):
        """Wrap a match-probability function into a knockout p_beat callback.

        Args:
            model_probs: Callable returning outcome probabilities for (a, b).

        Returns:
            Callable[[str, str], float]: p_beat(a, b), the probability a advances.
        """
        def p_beat(a, b):
            """Return the probability team a eliminates team b in a knockout tie.

            Args:
                a: One team.
                b: The other team.

            Returns:
                float: Probability a advances past the tie.
            """
            p = model_probs(a, b)
            return advance_prob_from_3way(p[Outcome.HOME], p[Outcome.DRAW])
        return p_beat

    def rf_pb(a, b):
        """Compute the hybrid RF outcome probabilities for a neutral-venue tie.

        Args:
            a: Home-listed team.
            b: Away-listed team.

        Returns:
            dict: Outcome probabilities keyed by Outcome for a vs b.
        """
        return rf26.match_probabilities_from_rows(
            pd.DataFrame(match_rows("qf", a, b, None, None, snaps26)))

    cond = {}
    for name, pb in (("Hybrid RF", beat(rf_pb)),
                     ("Dixon-Coles", beat(lambda a, b: dcs[2026].match_probabilities(a, b, neutral=True))),
                     ("Elo", beat(lambda a, b: elo26.match_probabilities(a, b, neutral=True)))):
        fc = solve_bracket(QF_PAIRS_2026, pb)
        cond[name] = {t: round(fc.p_champion[t], 4) for t in
                      sorted(fc.p_champion, key=lambda t: -fc.p_champion[t])}
    outr = Path("data/raw/wc2026_outrights.csv")
    if outr.exists():
        o = pd.read_csv(outr)
        inv = (1 / o["odds"]).sum()
        cond["Market (outrights)"] = {r.team: round((1 / r.odds) / inv, 4)
                                      for r in o.itertuples(index=False)}
    wc26["conditional_qf"] = cond
    wc26["qf_bracket"] = [" v ".join(p) for p in QF_PAIRS_2026]
    results["wc2026"] = wc26

    Path("paper").mkdir(exist_ok=True)
    Path("paper/results.json").write_text(json.dumps(results, indent=1, default=float))
    print("\nsaved -> paper/results.json")


if __name__ == "__main__":
    main()
