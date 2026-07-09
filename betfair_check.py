"""Betfair connectivity check (Milestone 5, read-only).

Logs in with your certificate, lists live soccer MATCH_ODDS markets (optionally filtered
by a text query, e.g. a team name), and prints the back/lay ladders for the first one.
No orders are placed. Requires BETFAIR_* creds + cert files in .env / certs/.

    python betfair_check.py
    python betfair_check.py --text "World Cup"
    python betfair_check.py --text "Brazil"
"""
from __future__ import annotations

import argparse

from wc_trader.adapters.betfair import BetfairVenue
from wc_trader.config import Secrets, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=None, help="text query to narrow markets (team/comp)")
    ap.add_argument("--max", type=int, default=10, help="max markets to list")
    args = ap.parse_args()

    cfg = load_config()
    venue = BetfairVenue(Secrets(), commission_rate=cfg.betfair.commission_rate,
                         markets=cfg.betfair.markets)

    try:
        venue.connect()
    except RuntimeError as e:
        print(f"Cannot connect: {e}")
        print("→ Fill BETFAIR_* in .env and place your cert/key under certs/ (see .env.example).")
        return

    print(f"Logged in. Balance: £{venue.balance():.2f}")
    markets = venue.list_markets({"text": args.text, "max_results": args.max})
    print(f"\n{len(markets)} MATCH_ODDS market(s)"
          + (f" matching '{args.text}'" if args.text else "") + ":")
    for m in markets:
        print(f"  {m.market_id}  {m.event_name}  (start {m.start_time})")

    if not markets:
        return

    m = markets[0]
    print(f"\nBack/lay ladders for: {m.event_name}")
    for r in m.runners:
        book = venue.get_order_book(r.selection_id)
        tag = r.outcome.value if r.outcome else "?"
        print(f"  {r.name:<20} [{tag:<4}]  best back {book.best_back}   best lay {book.best_lay}")


if __name__ == "__main__":
    main()
