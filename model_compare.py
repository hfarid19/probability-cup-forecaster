"""Model comparison entrypoint (Milestone 4): Elo vs Dixon-Coles on internationals.

    python model_compare.py
    python model_compare.py --eval-start 2014-01-01

Dixon-Coles is refit yearly on a trailing window; both models are scored on the same
out-of-sample matches. DC earns its place only if it beats Elo on log-loss.
"""
from __future__ import annotations

import argparse

from wc_trader.backtest.compare import compare_models
from wc_trader.data.results import fetch_results, load_results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-start", default="2018-01-01")
    ap.add_argument("--refit-days", type=int, default=365)
    args = ap.parse_args()

    fetch_results()
    df = load_results()
    print(f"Loaded {len(df):,} internationals. Comparing Elo vs Dixon-Coles "
          f"out-of-sample since {args.eval_start} (refit every {args.refit_days}d)...")

    rep = compare_models(df, eval_start=args.eval_start, refit_days=args.refit_days)

    print(f"\n{'=' * 60}\nElo vs Dixon-Coles — internationals since {rep.eval_start}\n{'=' * 60}")
    print(f"  eval matches            : {rep.n_eval:,}")
    print(f"  log-loss  Elo           : {rep.elo_log_loss:.4f}")
    print(f"  log-loss  Dixon-Coles   : {rep.dc_log_loss:.4f}")
    print(f"  log-loss  baseline      : {rep.baseline_log_loss:.4f}")
    print(f"  accuracy  Elo / DC      : {rep.elo_accuracy:.1%} / {rep.dc_accuracy:.1%}")
    diff = rep.elo_log_loss - rep.dc_log_loss
    print(f"\n  → Winner: {rep.winner}  (DC vs Elo: {diff:+.4f} log-loss)")
    if rep.winner == "Dixon-Coles":
        print("    Dixon-Coles earns its place as the primary model.")
    else:
        print("    Elo holds up — ship Elo, revisit DC with more features/tuning later.")


if __name__ == "__main__":
    main()
