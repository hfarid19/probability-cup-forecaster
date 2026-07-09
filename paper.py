"""Paper-trading entrypoint — runs the FULL pipeline against PaperVenue (simulated
fills) so the architecture is exercised end-to-end before Betfair is wired in.

    python paper.py

Milestone 1 deliverable: a runnable skeleton.
"""
from __future__ import annotations

from loguru import logger

from wc_trader.adapters.paper import PaperVenue
from wc_trader.config import Secrets, load_config
from wc_trader.engine.executor import Executor
from wc_trader.engine.risk import RiskManager
from wc_trader.model.elo import EloModel
from wc_trader.strategy.value import ValueStrategy


def run_once() -> None:
    cfg = load_config()
    _secrets = Secrets()  # loaded so missing creds surface early (used once Betfair is live)

    venue = PaperVenue(starting_balance=cfg.bankroll)
    # Seeded ratings for the synthetic England v France market (real ratings come from M2).
    # France is deliberately strong here so the model finds France underpriced vs the book
    # and the full back/risk/execute path fires in the demo.
    model = EloModel(ratings={"England": 1850.0, "France": 1925.0})
    strategy = ValueStrategy(cfg)
    risk = RiskManager(cfg, bankroll=cfg.bankroll)
    executor = Executor(venue, risk)

    logger.info(
        f"Paper trading | bankroll £{cfg.bankroll:.0f} | edge>={cfg.strategy.edge_threshold:.0%} "
        f"| {cfg.strategy.kelly_fraction:.0%}-Kelly | commission {cfg.betfair.commission_rate:.0%}"
    )

    for market in venue.list_markets():
        probs = model.match_probabilities("England", "France")
        logger.info(
            f"Model {market.event_name}: "
            + ", ".join(f"{o.value} {p:.1%}" for o, p in probs.items())
        )
        for runner in market.runners:
            book = venue.get_order_book(runner.selection_id)
            model_prob = probs[runner.outcome]
            signal = strategy.evaluate(runner.selection_id, model_prob, book, risk.bankroll)
            if signal:
                logger.info(
                    f"  EDGE {runner.name}: {signal.side.value} "
                    f"(model {signal.model_prob:.1%} vs market {signal.market_prob:.1%})"
                )
                executor.execute(signal)
            else:
                logger.info(f"  no edge on {runner.name}")

    logger.info(f"Open positions: {venue.positions()}")


if __name__ == "__main__":
    run_once()
