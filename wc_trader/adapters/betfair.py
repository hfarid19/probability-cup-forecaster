"""Betfair Exchange adapter (Milestone 5).

Wraps `betfairlightweight`:
  - non-interactive certificate login (headless bot auth),
  - Betting API for markets / books / orders / positions,
  - Account API for balance.

The library is imported lazily inside `connect()` so this module (and its pure helpers)
import fine without it. The tricky, bug-prone logic — Betfair's price-tick ladder and the
response→domain mapping — lives in module-level functions with unit tests, since a live
account can't be exercised here.

Interface note: the `Venue` methods take only `selection_id`, but Betfair scopes books and
orders by market. We keep a `selection_id → market_id` map, populated by `list_markets()`,
so you must call `list_markets()` before `get_order_book()` / `place_order()`.
"""
from __future__ import annotations

import re

from ..config import Secrets
from ..types import Market, OrderBook, Outcome, Position, PriceLevel, Runner, Side
from .base import Venue

# Betfair price increments (decimal-odds ladder): (from_inclusive, to_exclusive, tick).
TICK_BANDS = [
    (1.0, 2.0, 0.01), (2.0, 3.0, 0.02), (3.0, 4.0, 0.05), (4.0, 6.0, 0.1),
    (6.0, 10.0, 0.2), (10.0, 20.0, 0.5), (20.0, 30.0, 1.0), (30.0, 50.0, 2.0),
    (50.0, 100.0, 5.0), (100.0, 1000.0, 10.0),
]


def round_to_tick(price: float) -> float:
    """Snap a price to the nearest valid Betfair odds tick (off-ladder prices are rejected)."""
    price = min(1000.0, max(1.01, price))
    for lo, hi, inc in TICK_BANDS:
        if lo <= price < hi:
            steps = round((price - lo) / inc)
            return round(lo + steps * inc, 2)
    return price


def infer_outcome(runner_name: str, event_name: str) -> Outcome | None:
    """Tag a MATCH_ODDS runner as HOME/DRAW/AWAY using the 'Home v Away' event name."""
    if "draw" in runner_name.lower():
        return Outcome.DRAW
    parts = re.split(r"\s+v\s+|\s+vs\s+|\s+@\s+", event_name, flags=re.IGNORECASE)
    if len(parts) == 2:
        home, away = parts[0].strip().lower(), parts[1].strip().lower()
        rn = runner_name.strip().lower()
        if rn and (rn in home or home in rn):
            return Outcome.HOME
        if rn and (rn in away or away in rn):
            return Outcome.AWAY
    return None


def catalogue_to_market(cat) -> Market:
    """Map a betfairlightweight MarketCatalogue to our Market."""
    event_name = cat.event.name if getattr(cat, "event", None) else (cat.market_name or "")
    runners = [
        Runner(selection_id=int(r.selection_id), name=r.runner_name,
                outcome=infer_outcome(r.runner_name, event_name))
        for r in (cat.runners or [])
    ]
    start = cat.market_start_time.isoformat() if getattr(cat, "market_start_time", None) else ""
    return Market(market_id=cat.market_id, event_name=event_name,
                  market_type="MATCH_ODDS", start_time=start, runners=runners)


def market_book_to_orderbooks(book) -> dict[int, OrderBook]:
    """Map a MarketBook's runners to {selection_id: OrderBook} (back/lay ladders)."""
    out: dict[int, OrderBook] = {}
    for r in (book.runners or []):
        ex = getattr(r, "ex", None)
        backs = [PriceLevel(p.price, p.size) for p in (getattr(ex, "available_to_back", None) or [])] if ex else []
        lays = [PriceLevel(p.price, p.size) for p in (getattr(ex, "available_to_lay", None) or [])] if ex else []
        out[int(r.selection_id)] = OrderBook(selection_id=int(r.selection_id), backs=backs, lays=lays)
    return out


class BetfairVenue(Venue):
    SOCCER_EVENT_TYPE_ID = "1"

    def __init__(self, secrets: Secrets, commission_rate: float = 0.05,
                 markets: list[str] | None = None):
        self.secrets = secrets
        self.commission_rate = commission_rate
        self.markets = markets or ["MATCH_ODDS"]
        self._client = None
        self._filters = None
        self._sel_to_market: dict[int, str] = {}
        self._order_to_market: dict[str, str] = {}

    # ---- connection ----

    def connect(self):
        if self._client is not None:
            return self._client
        missing = [k for k in ("betfair_username", "betfair_password", "betfair_app_key",
                               "betfair_cert_path", "betfair_key_path")
                   if not getattr(self.secrets, k)]
        if missing:
            raise RuntimeError(f"Betfair credentials missing in .env: {missing}")
        import betfairlightweight
        from betfairlightweight import filters

        self._filters = filters
        self._client = betfairlightweight.APIClient(
            username=self.secrets.betfair_username,
            password=self.secrets.betfair_password,
            app_key=self.secrets.betfair_app_key,
            cert_files=(self.secrets.betfair_cert_path, self.secrets.betfair_key_path),
        )
        self._client.login()  # non-interactive cert login
        return self._client

    # ---- market data ----

    def list_markets(self, query: dict | None = None) -> list[Market]:
        client = self.connect()
        q = query or {}
        mf = self._filters.market_filter(
            event_type_ids=[self.SOCCER_EVENT_TYPE_ID],
            market_type_codes=self.markets,
            text_query=q.get("text"),
            competition_ids=q.get("competition_ids"),
            event_ids=q.get("event_ids"),
        )
        cats = client.betting.list_market_catalogue(
            filter=mf,
            market_projection=["EVENT", "MARKET_START_TIME", "RUNNER_DESCRIPTION"],
            max_results=q.get("max_results", 100),
        )
        markets = [catalogue_to_market(c) for c in cats]
        for m in markets:
            for r in m.runners:
                self._sel_to_market[r.selection_id] = m.market_id
        return markets

    def _market_id_for(self, selection_id: int) -> str:
        mid = self._sel_to_market.get(selection_id)
        if mid is None:
            raise KeyError(f"Unknown selection {selection_id}; call list_markets() first.")
        return mid

    def get_order_book(self, selection_id: int) -> OrderBook:
        client = self.connect()
        mid = self._market_id_for(selection_id)
        pp = self._filters.price_projection(price_data=["EX_BEST_OFFERS"])
        books = client.betting.list_market_book(market_ids=[mid], price_projection=pp)
        if not books:
            raise RuntimeError(f"No market book returned for {mid}")
        return market_book_to_orderbooks(books[0])[selection_id]

    # ---- orders ----

    def place_order(self, selection_id: int, side: Side, price: float, size: float) -> str:
        client = self.connect()
        mid = self._market_id_for(selection_id)
        instruction = self._filters.place_instruction(
            order_type="LIMIT",
            selection_id=selection_id,
            side=side.value,
            limit_order=self._filters.limit_order(
                size=round(size, 2),
                price=round_to_tick(price),
                persistence_type="LAPSE",
            ),
        )
        resp = client.betting.place_orders(market_id=mid, instructions=[instruction])
        report = resp.place_instruction_reports[0]
        if report.status != "SUCCESS":
            raise RuntimeError(f"Betfair rejected order: {report.status}/{report.error_code}")
        self._order_to_market[report.bet_id] = mid
        return report.bet_id

    def cancel(self, order_id: str) -> None:
        client = self.connect()
        mid = self._order_to_market.get(order_id)
        instruction = self._filters.cancel_instruction(bet_id=order_id)
        client.betting.cancel_orders(market_id=mid, instructions=[instruction])

    def positions(self) -> list[Position]:
        client = self.connect()
        current = client.betting.list_current_orders()
        agg: dict[int, list[float]] = {}  # selection -> [signed_matched, price*size, size]
        for o in current.orders:
            matched = float(getattr(o, "size_matched", 0) or 0)
            if matched <= 0:
                continue
            signed = matched if o.side == "BACK" else -matched
            avg = float(getattr(o, "average_price_matched", 0) or 0)
            sel = int(o.selection_id)
            if sel in agg:
                agg[sel][0] += signed
                agg[sel][1] += avg * matched
                agg[sel][2] += matched
            else:
                agg[sel] = [signed, avg * matched, matched]
        return [Position(sel, v[0], (v[1] / v[2] if v[2] else 0.0)) for sel, v in agg.items()]

    def balance(self) -> float:
        client = self.connect()
        funds = client.account.get_account_funds()
        return float(funds.available_to_bet_balance)
