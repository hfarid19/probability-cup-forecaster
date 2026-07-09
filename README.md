# wc-trader

**Implementing Groll, Ley, Schauberger & Van Eetvelde (2019) — "A hybrid random forest
to predict soccer matches in international tournaments" — and testing its merits live on
the 2026 FIFA World Cup.**

**The full replication paper is [`PAPER.md`](./PAPER.md)** — a section-by-section match
of the original (data, methods, leave-one-tournament-out comparison, tournament
simulations) applied to the **2022 World Cup** (complete) and the **2026 World Cup**
(through the Round of 16), with market benchmarks throughout. Regenerate every number
with `python paper_eval.py`.

The method is implemented faithfully (see [`GROLL_PLAN.md`](./GROLL_PLAN.md) for the
architecture and the exact deviation ledger), freezes all inputs at tournament start,
and is evaluated out-of-sample against strong baselines. A full soccer-trading stack
(Dixon-Coles, Elo, backtests, Betfair exchange adapter) was built along the way and
serves as the baseline + real-world-consequences layer; [`PLAN.md`](./PLAN.md)
documents that history.

> ⚠️ The trading layer involves real financial risk. Do **not** point it at live money;
> see the M3.5 finding below before interpreting any model as a betting edge.

## Results — 2026 World Cup, out-of-sample (96 matches through the Round of 16)

All features frozen at tournament start (2026-06-11); training on World Cups 1998–2022
only; nothing post-kickoff enters any model.

| Model | log-loss | Brier | accuracy |
|---|---|---|---|
| Hybrid RF (the paper) | 0.8559 | 0.5011 | 65.6% |
| **Dixon-Coles (baseline)** | **0.8152** | **0.4794** | **66.7%** |
| Elo | 0.9395 | 0.5313 | 63.5% |
| Base rate | 1.0628 | 0.6416 | 46.9% |
| *Market (consensus odds)* | *0.7906* | *0.4619* | *69.8%* |

**Findings so far:**

1. **The paper's method works** in the sense that matters most: it beats Elo and base
   rate comfortably, out-of-sample, on a tournament it never saw.
2. **But the "hybrid" adds nothing over its own core ingredient:** plain Dixon-Coles —
   which supplies the RF's ability covariates — leads the full RF by 0.041 log-loss.
   With truly fresh FIFA rankings the paired bootstrap puts that gap **just short of
   95% significance** (RF−DC diff +0.0408, CI [−0.0008, +0.0808], n=96): DC is at
   least as good, and probably better, on this tournament. (With the 630-day-stale
   rankings we first used, the gap was larger, +0.0477, and significant — fixing the
   covariate helped the RF, but not enough to catch its own baseline.)
3. **The paper's internal claim replicates:** ability parameters dominate the RF's
   variable importance (attack/defense fill the top ranks), exactly as Groll et al.
   report. The extra covariates (rank, GDP, population, host, confederation) carry
   little marginal signal; the market-value/squad covariates remain unsourceable (see
   the deviation ledger).
4. **Against the betting market** (OddsPortal consensus odds, scraped for all 96
   matches): the market is best on every metric, as expected. The hybrid RF is
   **significantly worse than the market** (paired diff +0.0653, CI [+0.0146,
   +0.1143]); Dixon-Coles is **statistically indistinguishable from the market**
   (+0.0245, CI [−0.0239, +0.0729]) on this sample. The paper's method does not
   approach market efficiency here; its own baseline nearly does.
5. Honest read: on a single 96-match tournament with a reduced covariate set, the
   sophisticated ensemble does not improve on the classical parametric model it wraps,
   and it trails the market significantly. That is a real replication observation —
   not an implementation failure (the RF still shows strong skill vs Elo/base rate).

**Data notes:**
- FIFA stopped publishing fetchable historical rankings after Sep 2024; the official
  11 June 2026 pre-tournament release was recovered from FIFA's public live-ranking
  API (`PrevRank`/`PrevPoints` = the last official release; the live fields include
  tournament results and would be lookahead). Chrome-DevTools network capture found
  the endpoint; see `wc_trader/data/fifa_rank.py`.
- Match odds come from OddsPortal's results pages (encrypted feed → scraped from the
  rendered DOM via headless Chrome + CDP; `scripts/scrape_wc2026_odds.py`, re-run it
  after each round). Consensus odds, overround ≈ 0.99 — effectively no-vig
  probabilities, the toughest fair benchmark.

Reproduce with `python groll_eval.py` (re-downloads results, so numbers update as the
tournament progresses; `--no-refresh` for the cached state).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python groll_eval.py        # the 2026 experiment: train WCs 1998–2022 → predict 2026
python paper_eval.py        # ALL paper numbers: LOO CV + 2022 + 2026 + simulations
python champion_forecast.py # who wins 2026? exact bracket forecast from the frozen models
python scripts/calibrate_news_layer.py        # lineup/form (β, γ) from 2018+2022
python scripts/market_edge_decomposition.py   # market-edge decomposition + news layer
python scripts/scrape_wc2026_odds.py   # refresh 2026 market odds (needs Chrome)
python scripts/scrape_wc2022_odds.py   # 2022 market odds (BetExplorer)
pytest                      # 39 tests

# Supporting research entrypoints
python model_compare.py     # Elo vs Dixon-Coles, out-of-sample (internationals)
python backtest.py          # Elo calibration vs base-rate baseline
python market_backtest.py --model dixon-coles   # model vs bookmaker closing line + P&L
python paper.py             # trading pipeline demo (paper venue, simulated fills)
python betfair_check.py     # Betfair connectivity (needs account + certs in .env)
```

## Layout

```
wc_trader/
  types.py            # domain types (Side, Market, OrderBook, Order, Position)
  config.py           # typed config (config.yaml) + secrets (.env)
  model/              # Elo, Dixon-Coles, and groll_rf.py (the paper: features + RF + double-Poisson)
  data/               # results, closing odds, FIFA rankings, World Bank, WC metadata
  backtest/           # metrics (incl. paired bootstrap), walk-forward harnesses, model-vs-market
  adapters/           # Venue interface + PaperVenue + BetfairVenue (cert login, orders)
  strategy/ engine/   # value strategy (edge + ¼-Kelly), risk caps, executor
groll_eval.py         # the 2026 replication experiment
GROLL_PLAN.md         # replication spec + faithfulness/deviation ledger
PLAN.md               # original trading-system design + milestone history
```

## Key prior findings (trading phase, kept for the record)

- **Dixon-Coles beats Elo** out-of-sample on internationals (log-loss 0.873 vs 0.913).
- **M3.5:** neither model beats sharp bookmaker closing lines in liquid leagues, and
  naive value-betting loses money (~−10% ROI) — a good model is not a betting edge.
- The Betfair adapter (cert login, live books, orders) is built and unit-tested but has
  never placed a real order; `live.py` stays guarded off.
