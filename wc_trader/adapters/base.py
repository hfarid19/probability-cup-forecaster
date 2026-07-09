"""Venue abstraction — all venue-specific code lives behind this single seam.

This lets the model, strategy, and risk layers stay identical whether we backtest,
paper-trade, or trade live on Betfair (or add another venue later).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import Market, OrderBook, Position, Side


class Venue(ABC):
    @abstractmethod
    def list_markets(self, query: dict | None = None) -> list[Market]:
        ...

    @abstractmethod
    def get_order_book(self, selection_id: int) -> OrderBook:
        ...

    @abstractmethod
    def place_order(self, selection_id: int, side: Side, price: float, size: float) -> str:
        """Submit an order (price = decimal odds, size = stake). Returns venue order id."""
        ...

    @abstractmethod
    def cancel(self, order_id: str) -> None:
        ...

    @abstractmethod
    def positions(self) -> list[Position]:
        ...

    @abstractmethod
    def balance(self) -> float:
        ...
