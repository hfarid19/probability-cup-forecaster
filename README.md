# wc-trader

**A replication of Groll, Ley, Schauberger & Van Eetvelde (2019) — "A hybrid random
forest to predict soccer matches in international tournaments" — tested on the 2022
World Cup (complete) and the 2026 World Cup (live, through the Round of 16).**

**The paper is [`PAPER.md`](./PAPER.md).** It matches the original section-by-section
(data, methods, leave-one-tournament-out comparison, tournament simulations) and adds
what the original couldn't have: a prospective test during a running tournament, a
market benchmark for every evaluated match, and a news-aware model extension.

## Headline results (details and honest caveats in the paper)

- The hybrid RF **beats Elo and base rate** out-of-sample, but **never clearly beats
  the Dixon-Coles ability model it is built on** — in leave-one-tournament-out CV they
  are statistically tied.
- The paper's mechanism claim **replicates robustly**: ability parameters dominate
  variable importance in every fit.
- **The market's edge is information, not modeling**: level with the frozen model on
  matchday 1, peaking on rotation-round matchday 3. Feeding the model official lineups
  + player form (calibrated on 2018+2022) recovers **market-level accuracy in the
  knockouts**; the matchday-3 residue (motivation, tactics) survives.
- Recorded in advance: the rating models make **Argentina** the 2026 favorite; the
  hybrid RF and the market say **France**. The final is 19 July 2026.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python paper_eval.py                          # regenerate ALL paper numbers (~5 min)
python scripts/calibrate_news_layer.py        # lineup/form coefficients (2018+2022)
python scripts/market_edge_decomposition.py   # §5.2 information decomposition
python scripts/scrape_wc2022_odds.py          # refresh 2022 market odds (BetExplorer)
python scripts/scrape_wc2026_odds.py          # refresh 2026 market odds (needs Chrome)
pytest                                        # 30 tests
```

Data (all public, cached under `data/raw/`): martj42 international results, FIFA
rankings (historical CSV + the official 2026-06-11 release recovered from FIFA's
live-ranking API), World Bank GDP/population, FIFA calendars and per-match lineups +
goal events, scraped closing odds.

## Layout

```
wc_trader/
  experiment.py       # freeze-protocol machinery: frames, snapshots, frozen models
  model/              # Elo, Dixon-Coles, groll_rf (the paper), lineup_adjust (news layer)
  data/               # results, FIFA rankings, World Bank, WC metadata, lineups
  backtest/           # metrics (log-loss/Brier/RPS/bootstrap), bracket solver, MC simulator
paper_eval.py         # the experiment: LOO CV + 2022 + 2026 + simulations -> paper/results.json
scripts/              # odds scrapers, news-layer calibration, market-edge decomposition
PAPER.md              # the paper
paper/                # results.json, news_coefficients.json
```

Re-run `paper_eval.py` (and the scrapers) after each 2026 round — every table updates.
