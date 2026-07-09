# Replicating Groll, Ley, Schauberger & Van Eetvelde (2019) — and testing it on the 2026 World Cup

**Paper:** "A hybrid random forest to predict soccer matches in international tournaments"
(Groll, Ley, Schauberger, Van Eetvelde, *Journal of Quantitative Analysis in Sports*, 2019).

**Project goal (reframed):** implement the paper's method faithfully, then evaluate its
merits *out-of-sample on a real-world event happening right now* — the 2026 FIFA World
Cup — against strong baselines (Dixon-Coles, Elo, base rate). This is an applied
replication: faithful model, real-world test (we do not re-run the paper's original
2002–2014→2018 experiment).

---

## 1. The paper's method (what "faithful" means)

1. **Unit of observation:** each match yields **two rows** — one per team — with the
   response = **goals scored by that team**.
2. **Hybrid part:** alongside classic covariates, each row carries **team ability
   parameters** estimated from a separate **time-weighted Poisson ranking model**
   (Ley/Van Eetvelde/Wenseleers abilities). These abilities are the paper's dominant
   predictor by variable importance.
3. **Learner:** a **random forest** regression on goals → expected goals λ for each
   team in each match.
4. **Match probabilities:** treat the two predicted λs as independent Poisson means →
   score grid → P(win/draw/loss).
5. **Tournament evaluation:** train on *past* World Cups only; predict the target
   World Cup.

## 2. Faithfulness ledger (exact deviations, stated up front)

| Paper element | Our implementation | Status |
|---|---|---|
| Two-rows-per-match, goals response | same | ✅ faithful |
| Ability parameters from time-weighted Poisson | our **Dixon-Coles** fit (time-weighted MLE, half-life 540d) supplies attack/defense abilities | ≈ faithful (DC adds a low-score correction; same family, same role) |
| Random forest (R `ranger`, ~5000 trees) | sklearn `RandomForestRegressor` (1000 trees) | ≈ faithful |
| Covariates: FIFA rank | historical FIFA ranking points + rank position | ✅ |
| Covariates: GDP per capita, population | World Bank API | ✅ |
| Covariates: host, confederation | static metadata | ✅ |
| Covariates: market value, squad age, legionnaires, CL/EL players, coach factors | **dropped — not sourceable historically** | ❌ documented gap |
| Training tournaments | WCs **1998–2022** (7 editions; paper used 4) | deviation (more data) |
| Independent double-Poisson for outcome probs | same | ✅ |
| Knockout scores | dataset records scores **incl. extra time**; we evaluate all matches and group-stage-only separately | documented |

The dropped covariates are real but secondary in the paper's own variable-importance
analysis; the **abilities — which dominate — we have in full**. A faithful *architecture*
with a documented covariate subset.

## 3. The 2026 test protocol (out-of-sample by construction)

- **Freeze date = tournament start (2026-06-11).** Abilities fit on matches strictly
  before it; FIFA rank = last release ≤ freeze; GDP/population = latest year ≤ 2025;
  Elo baseline frozen at the same date. Nothing after kickoff of the tournament enters
  any model.
- **Evaluate** on every 2026 WC match already played: log-loss, Brier, accuracy for
  **Hybrid RF vs Dixon-Coles vs Elo vs base rate**, all matches + group-only split.
- **Also check the paper's core internal claim:** ability parameters should dominate
  RF variable importance. We report the importance ranking.

## 4. Data sources

| Data | Source | Notes |
|---|---|---|
| Match results (train + eval) | martj42 international results | already in repo (`data/results.py`) |
| FIFA rankings 1992–2024 | Dato-Futbol `ranking_fifa_historical.csv` | name harmonization needed ("Korea Republic"→"South Korea", …) |
| FIFA rankings 2025–26 | official 2026-06-11 release recovered from FIFA's live-ranking API (`PrevPoints` = last official release; live fields are in-tournament → lookahead, never used) | ✅ staleness 0 days |
| GDP pc / population | World Bank API (`NY.GDP.PCAP.CD`, `SP.POP.TOTL`) | football→ISO3 map; England etc. → GBR |
| Hosts / confederations | static (`data/wc_meta.py`) | 2026 hosts: USA, Canada, Mexico |

## 5. Deliverables

- `wc_trader/data/fifa_rank.py`, `worldbank.py`, `wc_meta.py` — covariate loaders.
- `wc_trader/model/groll_rf.py` — feature builder + `HybridRF` + double-Poisson mapping.
- `groll_eval.py` — the experiment: train 1998–2022 → predict 2026, full report.
- Tests for all pure logic (as-of lookups, row building, probability mapping).
- Honest results section in README (whatever the numbers say).

## 6. Results (2026-07-08 — 96 matches through the Round of 16)

See `README.md` for the full table. Headlines: the hybrid RF **beats Elo and base rate**
out-of-sample (log-loss 0.856 vs 0.940 / 1.063) and the paper's variable-importance
claim **replicates** (abilities dominate). Plain **Dixon-Coles still leads the full RF**
(0.815 vs 0.856); with fresh freeze-date rankings the paired bootstrap gap is +0.0408,
95% CI [−0.0008, +0.0808] — just short of significance (it *was* significant, +0.0477,
under the initial 630-day-stale rankings; fixing the covariate closed part of the gap).
Against **market consensus odds** (0.791, scraped for all 96 matches): the RF is
significantly worse (+0.0653, CI [+0.0146, +0.1143]); Dixon-Coles is statistically
indistinguishable from the market (+0.0245, CI [−0.0239, +0.0729]).
Caveats: reduced covariate set, single tournament. Re-run `python groll_eval.py` (and
`python scripts/scrape_wc2026_odds.py`) as more matches complete.

## 7. Relationship to the existing codebase

The trading stack (Betfair venue, strategy, risk) is **retained** — it becomes the
"real-world consequences" layer: if the replicated model shows skill, it can be
paper-traded through the existing pipeline. The M3.5 finding (a good model ≠ beating
the market) remains the sober backdrop for any betting interpretation.
