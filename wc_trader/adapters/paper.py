"""Paper venue: simulates fills locally so the full pipeline runs without real money.

For Milestone 1 it also generates a synthetic market, so `python paper.py` works
end-to-end before the Betfair adapter is wired up. Later, feed it BetfairVenue's live
*data* (order books) while keeping fills simulated — that's the real paper-trading mode.
"""
from __future__ import annotations

from loguru import logger

from ..types import Market, Order, OrderBook, Outcome, Position, PriceLevel, Runner, Side
from .base import Venue


class PaperVenue(Venue):
    def __init__(self, starting_balance: float = 1000.0):
        self._balance = starting_balance
        self._positions: dict[int, Position] = {}
        self._orders: dict[str, Order] = {}
        self._next_id = 1
        self._market = self._synthetic_market()
        # Synthetic back/lay ladders keyed by selection_id (decimal odds).
        # Implied probs here are deliberately a bit off so the Elo model finds an edge.
        self._books = {
            1: OrderBook(1, backs=[PriceLevel(2.50, 200)], lays=[PriceLevel(2.56, 200)]),
            2: OrderBook(2, backs=[PriceLevel(3.40, 150)], lays=[PriceLevel(3.50, 150)]),
            3: OrderBook(3, backs=[PriceLevel(3.20, 180)], lays=[PriceLevel(3.30, 180)]),
        }

    @staticmethod
    def _synthetic_market() -> Market:
        return Market(
            market_id="1.SYNTH",
            event_name="England v France",
            market_type="MATCH_ODDS",
            start_time="2026-06-26T18:00:00Z",
            runners=[
                Runner(1, "England", Outcome.HOME),
                Runner(2, "France", Outcome.AWAY),
                Runner(3, "The Draw", Outcome.DRAW),
            ],
        )

    def list_markets(self, query: dict | None = None) -> list[Market]:
        return [self._market]

    def get_order_book(self, selection_id: int) -> OrderBook:
        return self._books[selection_id]

    def place_order(self, selection_id: int, side: Side, price: float, size: float) -> str:
        order_id = f"paper-{self._next_id}"
        self._next_id += 1
        self._orders[order_id] = Order(self._market.market_id, selection_id, side, price, size, order_id)
        # Naive instant fill at the requested price (good enough for the skeleton).
        signed = size if side == Side.BACK else -size
        pos = self._positions.get(selection_id)
        if pos:
            pos.net_stake += signed
        else:
            self._positions[selection_id] = Position(selection_id, signed, price)
        logger.info(f"[PAPER] filled {side.value} {size:.2f} @ {price:.2f} on sel {selection_id} -> {order_id}")
        return order_id

    def cancel(self, order_id: str) -> None:
        self._orders.pop(order_id, None)

    def positions(self) -> list[Position]:
        return list(self._positions.values())

    def balance(self) -> float:
        return self._balance
