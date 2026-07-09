# World Cup Algorithmic Trading — Build Plan

> **Project reframed (July 2026):** the primary goal is now a faithful implementation of
> **Groll et al. (2019)** tested out-of-sample on the 2026 World Cup — see
> [`GROLL_PLAN.md`](./GROLL_PLAN.md) and the results in `README.md`. This document
> remains the design + milestone history of the trading stack, which now serves as the
> baseline and infrastructure layer for that replication.

A directional, model-based trading system for World Cup soccer markets on the
**Betfair Exchange**, written in Python. "Directional" means: we build our own estimate
of true outcome probabilities, compare it to the market's implied probability, and
take a position only when our edge exceeds costs.

Betfair is a true peer-to-peer **exchange**: you can **back** (bet for) *and* **lay**
(bet against) an outcome, trade in and out before resolution ("green up"), and you pay
**commission on net winnings** rather than crossing a bid/ask spread. It has the
deepest soccer liquidity anywhere and — importantly — **downloadable historical
order-book data** for realistic backtesting.

> Status: **planning document**. No trading code written yet. Review and adjust,
> then we build in the order below.

---

## 0. The thesis (where the money comes from)

```
edge = P_model(outcome) − P_market(outcome)
```

We bet when `edge` is large enough to overcome fees + slippage + model error.
Everything in this system exists to (a) make `P_model` accurate and well-calibrated,
(b) measure `P_market` precisely, and (c) size/execute so that a real edge survives
into realized PnL.

**If the model is not better-calibrated than the market, nothing else matters.**
That's why we build and validate the model *before* writing a single order.

---

## 1. Legal / account reality check (do this first)

- Betfair Exchange access requires a **funded Betfair account** + an **Application Key**
  (free "delayed" key for read-only/dev, live key for real-time + betting) obtained
  through Betfair's developer program.
- **Bot login = non-interactive certificate login**: you register an SSL client
  certificate so the system can authenticate headless (vs. the interactive username/
  password flow). Set this up early — it's the most common onboarding snag.
- **Geo/KYC is tied to your account and funding source, not just IP.** Betfair is
  region-restricted (not available in the US, among others); a VPN does not solve
  account-level KYC or withdrawal checks. Confirm you can fund *and withdraw* cleanly
  before putting real size on.
- **Action items:** open + verify account → get app key → register the cert for
  non-interactive login → confirm a manual bet + cash-out works → only then automate.

---

## 2. Architecture

```
          ┌──────────────┐
          │ Historical   │  international match results (free datasets)
          │ results data │
          └──────┬───────┘
                 │ fit
          ┌──────▼───────┐
          │   MODEL      │  Elo baseline  +  Dixon-Coles goals model
          │ P_model(out) │  → calibrated win/draw/loss & tournament probs
          └──────┬───────┘
                 │
   live prices   │           ┌─────────────┐
 ┌────────────┐  │           │   RISK      │ bankroll, exposure caps,
 │  Betfair   │──┼──────────▶│ kill-switch │ max position, drawdown halt
 │  adapter   │  │           └──────┬──────┘
 └────────────┘  ▼                  │
          ┌──────────────┐          │
          │  STRATEGY    │  edge = P_model − P_market
          │ value + Kelly│──────────┤ back/lay sized signal
          └──────┬───────┘          │
                 ▼                  ▼
          ┌──────────────┐   ┌─────────────┐
          │  EXECUTOR    │──▶│  Betfair    │  (paper: simulated fills)
          │ orders/fills │   │  Exchange   │  (live:  real bets)
          └──────────────┘   └─────────────┘
```

Key design rule: **the venue is an interface.** All Polymarket-specific code lives
behind one adapter implementing a small abstract API, so we can backtest, paper-trade,
or swap to Betfair/Kalshi without touching the model or strategy.

### Adapter interface (the seam)

```python
class Venue(ABC):
    def list_markets(self, query) -> list[Market]: ...
    def get_order_book(self, selection_id) -> OrderBook: ...   # back/lay ladders
    def place_order(self, selection_id, side, price, size) -> OrderId: ...  # side ∈ {BACK, LAY}
    def cancel(self, order_id) -> None: ...
    def positions(self) -> list[Position]: ...
    def balance(self) -> Decimal: ...
```
Note the `side ∈ {BACK, LAY}` and decimal-odds prices — the interface is built around
exchange semantics so the strategy can express "bet against" directly.

Three implementations over time:
1. `PaperVenue` — wraps a real venue's *data* (incl. Betfair's stream) but simulates
   fills locally against the live order book.
2. `BetfairVenue` — real bets via `betfairlightweight` (Betting API + Exchange Stream).
3. (later) `KalshiVenue` / others — if we ever want a second venue or cross-venue arb.

---

## 3. The model (the actual edge)

### 3a. Baseline: Elo ratings
- Each national team has a rating; update after every match by result + margin.
- Map rating difference → win/draw/loss probabilities.
- Pros: simple, robust, hard to overfit. Use as the **benchmark** every fancier model
  must beat.

### 3b. Primary: Dixon-Coles goals model
- Estimate per-team **attack** and **defense** strengths + home advantage.
- Model each team's goals as Poisson; Dixon-Coles adds a low-score correlation
  correction (the classic fix for soccer's draw/0-0 structure).
- Simulate the match → score distribution → P(win/draw/loss), and Monte-Carlo the
  bracket → P(team reaches QF / wins cup) for the futures markets.

### 3c. Calibration (non-negotiable)
- Split data by time (train on past, test on later) — **never** random split.
- Score with **log-loss** and **Brier score**; plot a **reliability curve** (predicted
  vs actual frequency). A model that's accurate but mis-calibrated will bleed money.
- Compare model probabilities to *historical closing market prices*: the market is a
  strong baseline. If we can't beat the closing line, we don't have an edge.

### Data sources (all free, no scraping needed)
- International results datasets (Kaggle "International football results 1872–present").
- FIFA rankings for priors.
- Historical Polymarket/closing odds for the market benchmark (collected live as we go).

---

## 4. Strategy & sizing

- **Entry:** take a position when `|edge| > threshold`, where threshold covers
  **Betfair commission on net winnings** (market-dependent, typically ~2–5%) +
  expected slippage on the back/lay ladder + a model-uncertainty buffer (start
  conservative, e.g. 3–5%). Note Betfair's cost model is commission-on-profit, **not**
  a per-trade spread — size the threshold off the implied-prob equivalent.
- **Back vs lay:** when `P_model > P_market` → **back**; when `P_model < P_market`
  → **lay**. The exchange lets us take whichever side is mispriced, and trade out
  ("green up") to lock profit before resolution if the price converges.
- **Sizing:** **fractional Kelly** (¼-Kelly to start) on the per-market edge, then
  clamp by hard risk caps. Full Kelly is too aggressive given model error.
- **Exits:** close/reduce (or green up) when edge decays below threshold, on news, or
  at resolution.
- **Correlation awareness:** group-stage and bracket markets are correlated — cap
  *aggregate* exposure to a single team/outcome, not just per-market.

---

## 5. Risk management (the thing that keeps you solvent)

- `max_position_per_market`, `max_exposure_per_team`, `max_total_deployed`.
- **Daily/total drawdown kill-switch** → flatten and halt on breach.
- Order sanity checks: never cross the book by more than X, min price tick, max size.
- Heartbeat + reconnect logic for the data feed; **fail closed** (stop trading) on
  stale data or API errors.
- Full audit log of every signal, order, fill, and rejection.

---

## 6. Validation ladder (do NOT skip steps)

1. **Backtest** — replay **Betfair historical data** (downloadable order-book/price
   history — a major advantage over Polymarket) vs actual results; measure edge,
   calibration, and simulated PnL with realistic commission/slippage.
2. **Paper trade** — live Polymarket data, **simulated fills**, for *weeks*. Confirm
   the model's live edge matches backtest expectations.
3. **Live micro** — real orders at trivial size; confirm fills, fees, and slippage
   match the paper model.
4. **Scale gradually** — increase size only while realized calibration holds.

---

## 7. Build order (milestones)

| # | Milestone | Deliverable |
|---|-----------|-------------|
| 1 | Repo scaffold + config + Venue interface | runnable skeleton ✅ |
| 2 | Historical data loader | clean results dataset ✅ (~50k internationals) |
| 3 | Elo baseline + backtest harness | calibration report ✅ (+12–13% log-loss skill) |
| 3.5 | Model vs market closing line + P&L sim | ✅ neither beats the sharp line (Elo −10.3% / DC −9.4% ROI) |
| 4 | Dixon-Coles model | ✅ beats Elo (0.873 vs 0.913 log-loss) — primary model |
| 5 | `BetfairVenue` (cert login + Betting/Account API) | ✅ live markets, books, orders, positions, balance (pure logic unit-tested; live path needs certs) |
| 6 | Strategy (edge + Kelly) + risk caps | signals from live data |
| 7 | `PaperVenue` + paper loop | weeks of paper PnL |
| 8 | Live executor (tiny size) | real fills, audited |

We are here: **#1–#5 + #3.5 complete.** Dixon-Coles is the primary model and the Betfair
adapter is built (cert login, live books, orders — pure logic unit-tested, live path
awaits your account/certs via `betfair_check.py`). The pivotal finding from #3.5 stands:
a well-calibrated model does **not** beat sharp closing lines in liquid leagues, and
naive value-betting loses money. The edge — if it exists — is in *less efficient* markets
(lower leagues, in-play, World Cup props) and in *selective, shrunk* betting. Next: a live
loop (#6) polling Betfair into the model+strategy with **simulated fills** (#7) — which
also yields the first model-vs-market test on *internationals* (real Betfair odds).

(The alternative-data / Reddit-sentiment angle was explored and **dropped** — the market
reacts to the broadcast faster than the crowd types, Betfair's in-play bet delay erodes
any reaction edge, and pre-match sentiment is too noisy/biased to trust. Not worth the
complexity.)

---

## 8. Open questions for you

- **Capital & risk tolerance** — rough bankroll and max acceptable drawdown? (Drives
  Kelly fraction and caps.)
- **Market focus** — match winners (3-way), tournament futures (winner/advance), or
  both? Futures need the bracket Monte-Carlo; match markets are simpler to start.
- **Latency** — directional/value trading is not latency-sensitive (minutes is fine),
  so we can poll rather than build streaming infra first. Agree?
- **Time horizon** — building/validating a calibrated model properly takes weeks of
  paper trading. Comfortable with that before real money?

---

## 9. Tech stack

- Python 3.11+, `uv` or `poetry` for deps.
- `betfairlightweight` (Betfair Betting API + Exchange Stream + historical data parsing),
  `pandas`/`numpy`/`scipy` (model), `pydantic` (config/typed models), `pytest` (tests),
  `loguru` (audit logging).
- Local SQLite/Parquet for results, prices, fills, and the audit trail.
