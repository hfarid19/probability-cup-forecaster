"""Backtest entrypoint (Milestone 3): fit Elo walk-forward and print a calibration report.

    python backtest.py                       # all internationals since 2010
    python backtest.py --eval-start 2014-01-01
    python backtest.py --world-cup           # evaluate on World Cup matches only
    python backtest.py --tune                # grid-search Elo hyperparameters

Downloads the dataset on first run.
"""
from __future__ import annotations

import argparse

from wc_trader.backtest.simulator import BacktestReport, run_elo_backtest, tune_elo
from wc_trader.data.results import fetch_results, load_results

WORLD_CUP = ("FIFA World Cup",)


def print_report(report: BacktestReport, title: str) -> None:
    skill = report.skill_vs_baseline
    verdict = "✅ beats baseline" if skill > 0 else "❌ NO skill over baseline"
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")
    print(f"  params            : {report.params}")
    print(f"  eval matches      : {report.n_eval}"
          + (f"  (since {report.eval_start})" if report.eval_start else ""))
    if report.tournaments:
        print(f"  tournaments       : {', '.join(report.tournaments)}")
    print(f"  log-loss (model)  : {report.log_loss:.4f}")
    print(f"  log-loss (baseline): {report.baseline_log_loss:.4f}")
    print(f"  skill vs baseline : {skill:+.1%}   {verdict}")
    print(f"  Brier score       : {report.brier:.4f}")
    print(f"  accuracy (top-1)  : {report.accuracy:.1%}")
    print("\n  Reliability (calibration) — predicted vs observed by bin:")
    print(f"    {'bin':>12} {'n':>7} {'predicted':>10} {'observed':>10} {'gap':>8}")
    for b in report.reliability:
        gap = b.observed_freq - b.mean_predicted
        print(f"    {b.lo:.1f}-{b.hi:.1f}{'':>5} {b.n:>7} "
              f"{b.mean_predicted:>9.1%} {b.observed_freq:>9.1%} {gap:>+7.1%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-start", default="2010-01-01", help="ISO date; evaluate matches on/after this")
    ap.add_argument("--world-cup", action="store_true", help="evaluate on FIFA World Cup matches only")
    ap.add_argument("--tune", action="store_true", help="grid-search Elo hyperparameters")
    args = ap.parse_args()

    fetch_results()
    df = load_results()
    print(f"Loaded {len(df):,} international matches ({df.date.min().date()} → {df.date.max().date()})")

    tournaments = WORLD_CUP if args.world_cup else None
    scope = "FIFA World Cup matches" if args.world_cup else "all internationals"

    if args.tune:
        best, all_reports = tune_elo(df, eval_start=args.eval_start, tournaments=tournaments)
        print(f"\nGrid search over {len(all_reports)} configs — best by out-of-sample log-loss:")
        print_report(best, f"BEST Elo config — {scope} since {args.eval_start}")
    else:
        report = run_elo_backtest(df, eval_start=args.eval_start, tournaments=tournaments)
        print_report(report, f"Elo walk-forward — {scope} since {args.eval_start}")


if __name__ == "__main__":
    main()
