"""Live trading entrypoint — intentionally GUARDED.

Refuses to run until the real Betfair adapter (M5), the backtest (M3), and a paper-
trading track record (M7) exist. Going live before the validation ladder is complete is
how money is lost. See PLAN.md §6 (validation ladder) and §7 (build order).
"""
from __future__ import annotations

import sys


def main() -> None:
    print("Live trading is intentionally disabled in the scaffold.")
    print("Complete Milestones 2-7 first: model -> backtest -> Betfair adapter -> paper track record.")
    print("See PLAN.md §6 (validation ladder) and §7 (build order).")
    sys.exit(1)


if __name__ == "__main__":
    main()
