"""Model-vs-market backtest entrypoint (Milestone 3.5).

Downloads club-league results + closing odds, runs walk-forward Elo against the closing
line, and simulates value betting after commission.

    python market_backtest.py
    python market_backtest.py --eval-start 2023-01-01 --edge 0.05
"""
from __future__ import annotations

import argparse

from wc_trader.backtest.market import run_dc_vs_market, run_elo_vs_market
from wc_trader.data.odds import DEFAULT_LEAGUES, DEFAULT_SEASONS, load_odds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["elo", "dixon-coles"], default="elo")
    ap.add_argument("--eval-start", default="2022-07-01", help="evaluate matches on/after this date")
    ap.add_argument("--edge", type=float, default=0.05, help="min EV at offered odds to bet")
    ap.add_argument("--commission", type=float, default=0.02)
    args = ap.parse_args()

    print("Downloading club-league results + closing odds (cached after first run)...")
    df = load_odds(DEFAULT_SEASONS, DEFAULT_LEAGUES)
    print(f"Loaded {len(df):,} matches with odds "
          f"({df.date.min().date()} → {df.date.max().date()}, leagues: {sorted(df.league.unique())})")
    print(f"Avg overround (vig): {df.overround.mean():.3f}")

    if args.model == "dixon-coles":
        rep = run_dc_vs_market(df, eval_start=args.eval_start, edge_threshold=args.edge,
                               commission=args.commission)
    else:
        rep = run_elo_vs_market(df, eval_start=args.eval_start, edge_threshold=args.edge,
                                commission=args.commission)

    print(f"\n{'=' * 64}\n{args.model} vs closing line — since {args.eval_start}\n{'=' * 64}")
    print(f"  eval matches        : {rep.n_eval:,}")
    print(f"  log-loss (model)    : {rep.model_log_loss:.4f}")
    print(f"  log-loss (market)   : {rep.market_log_loss:.4f}")
    verdict = "✅ beats the line" if rep.beats_line else "❌ does NOT beat the line (expected — it's sharp)"
    print(f"  vs closing line     : {rep.model_log_loss - rep.market_log_loss:+.4f}   {verdict}")
    print(f"\n  Value betting (EV > {rep.edge_threshold:.0%} at offered odds, {rep.commission:.0%} commission):")
    print(f"    bets placed       : {rep.n_bets:,}")
    print(f"    hit rate          : {rep.hit_rate:.1%}")
    print(f"    total staked      : {rep.total_staked:,.0f} units")
    print(f"    profit            : {rep.profit:+,.1f} units")
    roi_verdict = "✅ profitable" if rep.roi > 0 else "❌ unprofitable"
    print(f"    ROI               : {rep.roi:+.1%}   {roi_verdict}")
    print("\n  Reminder: bookmaker odds carry vig AND we charge commission — this is a")
    print("  conservative lower bound vs. real Betfair exchange prices.")


if __name__ == "__main__":
    main()
