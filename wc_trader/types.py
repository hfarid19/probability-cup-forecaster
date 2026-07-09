"""Core domain types — venue-agnostic, built around exchange (back/lay) semantics."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Side(str, Enum):
    BACK = "BACK"   # bet FOR an outcome
    LAY = "LAY"     # bet AGAINST an outcome (exchange-only)


class Outcome(str, Enum):
    HOME = "HOME"
    DRAW = "DRAW"
    AWAY = "AWAY"


@dataclass(frozen=True)
class Runner:
    """A selection within a market (e.g. 'England' in a Match Odds market)."""
    selection_id: int
    name: str
    outcome: Outcome | None = None


@dataclass
class Market:
    market_id: str
    event_name: str          # e.g. "England v France"
    market_type: str         # e.g. "MATCH_ODDS"
    start_time: str          # ISO8601
    runners: list[Runner] = field(default_factory=list)


@dataclass(frozen=True)
class PriceLevel:
    price: float             # decimal odds
    size: float              # stake available at this price


@dataclass
class OrderBook:
    selection_id: int
    backs: list[PriceLevel]  # available-to-back ladder, best first
    lays: list[PriceLevel]   # available-to-lay ladder, best first

    @property
    def best_back(self) -> float | None:
        return self.backs[0].price if self.backs else None

    @property
    def best_lay(self) -> float | None:
        return self.lays[0].price if self.lays else None


@dataclass
class Order:
    market_id: str
    selection_id: int
    side: Side
    price: float             # decimal odds
    size: float              # stake
    order_id: str | None = None


@dataclass
class Position:
    selection_id: int
    net_stake: float         # signed exposure (+back / -lay)
    avg_price: float
