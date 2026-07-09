# A hybrid random forest to predict soccer matches in international tournaments: a replication on the 2022 and 2026 FIFA World Cups

*A section-by-section replication of Groll, Ley, Schauberger & Van Eetvelde (2019),
"A hybrid random forest to predict soccer matches in international tournaments,"
Journal of Quantitative Analysis in Sports 15(4), applied to the 2022 FIFA World Cup
(complete, retrospective) and the 2026 FIFA World Cup (prospective, through the Round
of 16 as of 8 July 2026).*

All numbers are reproducible: `python paper_eval.py` regenerates `paper/results.json`
from public data (20,000 tournament simulations, fixed seeds). Model code:
`wc_trader/model/groll_rf.py`; simulator: `wc_trader/backtest/tournament.py`.

---

## 1. Introduction

Forecasting the outcomes of international soccer tournaments has produced two largely
separate traditions: *regression-based* approaches that model goals as a function of
observable covariates (economic strength, FIFA rank, squad structure), and
*ranking-based* approaches that estimate latent team abilities from historical match
results (Poisson-family strength models in the line of Maher 1982 and Dixon & Coles
1997; Elo-type ratings). Groll et al. (2019) proposed joining the two: a **random
forest** trained on tournament matches whose covariate set is augmented with **ability
parameters** estimated separately by a time-weighted Poisson ranking model. On World
Cups 2002–2014 the hybrid outperformed both parent approaches, and its 2018 forecast
became one of the more widely cited applications of machine learning in sports
prediction.

This paper asks the question every applied method must eventually face: **do the
paper's merits hold up out-of-sample, on tournaments it has never seen, against strong
baselines and against the betting market?** We implement the method faithfully (§2–§3
document every deviation), reproduce its methodology one-to-one — the
leave-one-tournament-out comparison of §4 and the tournament forecast of §5 — and
apply it to the 2022 World Cup retrospectively and to the 2026 World Cup *while it is
being played*.

Relative to the original we add two things it could not have: a genuinely prospective
test, and a market benchmark built from consensus betting odds for every evaluated
match.

## 2. Data

As in the original, the unit of observation is a **(team, match)** pair: every match
contributes two rows, the response being the goals scored by the row's team. Training
data cover the **seven World Cups 1998–2022** (448 matches, 896 observations; the
original used four, 2002–2014).

### 2.1 Covariates

All covariates are frozen at each tournament's start date — nothing measured after
kickoff of the opening match enters any model.

**Ability parameters (the "hybrid" ingredient).** Attack and defense strengths from a
time-weighted Poisson goals model fit on all internationals in the 8 years before the
tournament (half-life 540 days; ~49,500-match database, 1872–present). Following Dixon
& Coles (1997) the fit includes a low-score dependence correction; this stands in for
the bivariate-Poisson abilities of Ley, Van de Wiele & Van Eetvelde (2019) used in the
original — same family, same role, same time-weighting idea.

**Sportive covariates.** FIFA World Ranking points and rank position at the last
official release before the tournament. (Sourcing note: FIFA's fetchable ranking
archive ends September 2024; the official 11 June 2026 pre-tournament release was
recovered from the `PrevPoints` field of FIFA's live-ranking API — the last *official*
release, verified against FIFA's published top-ranked team — because the live fields
update during the tournament and would leak outcomes.)

**Economic covariates.** GDP per capita and population (World Bank, latest year
strictly before the tournament; UK constituent teams share GBR values).

**Home advantage.** Host indicator (all three 2026 co-hosts flagged) and confederation
(one-hot, 7 levels).

**Not sourceable — dropped (deviation).** The original also used squad market value,
number of Champions/Europa League players, legionnaires, mean age, and coach covariates
(age, tenure, nationality). Historical, tournament-dated versions are not publicly
retrievable and are omitted. The original's own variable-importance analysis ranks the
ability parameters far above these; §5 verifies that this importance ordering
replicates. Three team-tournament covariate cells are missing (Serbia's FIFA entry in
1998/2006 under prior names; North Korea GDP 2010) and are median-imputed.

### 2.2 Response and match universe

Goals in the recorded final score. Knockout scores include extra time (dataset
convention); ties decided on penalties count as draws in the 3-way evaluation, and §5
reports group-stage and knockout splits separately.

### 2.3 Market data (new relative to the original)

Average 1X2 odds for **all 64 matches of 2022** (BetExplorer archive; mean overround
1.045) and **all 96 matches of 2026 played so far** (OddsPortal; mean overround 0.994),
plus outright winner odds for the remaining 2026 field (OddsPortal, 8 July 2026). Odds
are de-vigged by proportional normalization. Scrapers:
`scripts/scrape_wc2022_odds.py`, `scripts/scrape_wc2026_odds.py`.

**Information sets (read before comparing anything).** These are *closing* odds: they
embody everything known at kickoff. Our forecasters occupy explicit tiers: **T0** =
frozen at tournament start (the paper's protocol: hybrid RF, Dixon-Coles, Elo in
Tables 5.1/5.4); **T1** = T0 + tournament *results* to date (the "updated" variants of
§5.2); **T2** = T1 + official pre-kickoff lineups and prior-match events (the
news-adjusted models of §5.2). The market sits at **T2⁺** — a strict superset of T2
(injuries, motivation, tactical intent, order flow). Model-vs-market rows in Tables
5.1/5.4 are therefore *deliberately asymmetric* (T0 vs T2⁺): they measure the paper's
frozen-forecast protocol against the best-informed benchmark, so a market win there is
expected and a model win would be remarkable. The only approximately like-for-like
model-vs-market comparisons in this paper are the T2 news models of §5.2 — and even
those hold strictly less information than the closing line, so any parity claim is
conservative.

## 3. Methods

### 3.1 Random forests

`sklearn.ensemble.RandomForestRegressor`, 1,000 trees, minimum leaf size 5, one-third
of features per split (regression convention; the original used R `ranger` with ~5,000
trees — differences at this scale are negligible). 28 features after one-hot encoding.

### 3.2 Rating baselines

Where the original compared against Poisson regression, we field two baselines from
the rating tradition, both frozen at tournament start: (i) the **Dixon-Coles model
itself** — the hybrid's own ability source used directly as a forecaster — and (ii) an
**Elo rating** (K = 20, home advantage suppressed at neutral venues, empirically
calibrated draw model). The Dixon-Coles baseline is the decisive one: any lift the
hybrid shows over it measures exactly what the covariates and the forest add beyond
the abilities.

### 3.3 Ranking methods (ability estimation)

Attack/defense parameters estimated by weighted maximum likelihood with an L2 penalty;
teams with fewer than 8 rated matches in the window fall back to league-average
ability.

### 3.4 The hybrid random forest

Exactly as in the original: the forest regresses goals scored on the row team's and
the opponent's covariate vectors; a fixture yields expected goals (λ̂_A, λ̂_B); match
outcome probabilities follow from two independent Poissons over a 0–10 grid; tournament
probabilities from Monte-Carlo simulation (§5) with group tables (points, goal
difference, goals; residual ties random), FIFA's third-place slot-allocation
constraints solved per simulation (2026), and extra time/penalties as a coin flip.

## 4. Comparing the methods: leave-one-tournament-out

Each World Cup 1998–2022 is held out in turn; the hybrid RF is trained on the other
six tournaments, and all methods predict the held-out tournament. We report the
original's criteria — classification rate, average likelihood of the observed outcome,
ranked probability score (RPS) — plus log-loss, pooled over all 448 held-out matches.

**Table 4.1 — pooled leave-one-tournament-out performance (448 matches; best in bold).**

| Method | Class. rate ↑ | Likelihood ↑ | RPS ↓ | Log-loss ↓ |
|---|---|---|---|---|
| Hybrid RF | **0.554** | **0.4240** | 0.1950 | **0.9632** |
| Dixon-Coles | 0.545 | 0.4203 | **0.1942** | 0.9643 |
| Elo | 0.542 | 0.4118 | 0.2012 | 0.9867 |

**Table 4.2 — RPS by held-out tournament (best in bold).**

| Held-out WC | Hybrid RF | Dixon-Coles | Elo |
|---|---|---|---|
| 1998 | 0.1737 | **0.1680** | 0.1834 |
| 2002 | 0.2103 | **0.2036** | 0.2184 |
| 2006 | 0.1794 | **0.1681** | 0.1816 |
| 2010 | 0.1937 | **0.1870** | 0.1945 |
| 2014 | **0.1872** | 0.1956 | 0.2144 |
| 2018 | **0.1995** | 0.2025 | 0.2073 |
| 2022 | 0.2211 | 0.2346 | **0.2087** |

**Reading.** Both goal-based methods clearly beat Elo, but the original's headline
result — the hybrid outperforming the ranking method — does **not** clearly replicate:
pooled over seven tournaments the hybrid and its own ability source are statistically
tied (the hybrid is a hair better on three criteria, worse on RPS). The per-tournament
pattern is more interesting: Dixon-Coles wins every edition from 1998–2010, the hybrid
wins 2014–2022. Whether that reflects a real regime change (covariates growing more
informative) or noise cannot be settled with seven tournaments.

## 5. Modeling the 2022 and 2026 World Cups

### 5.1 The 2022 World Cup (retrospective, fully out-of-sample)

Hybrid RF trained on 1998–2018 only; everything frozen at 20 November 2022.

**Table 5.1 — match-level evaluation, all 64 matches.**

| Method | Class. rate ↑ | Likelihood ↑ | RPS ↓ | Log-loss ↓ |
|---|---|---|---|---|
| Hybrid RF | 0.516 | 0.3925 | 0.2211 | 1.0401 |
| Dixon-Coles | 0.438 | 0.3935 | 0.2346 | 1.1164 |
| **Elo** | **0.547** | 0.4164 | 0.2087 | **0.9897** |
| Market (de-vigged avg. odds) | 0.531 | **0.4301** | **0.2078** | 1.0013 |

Paired log-loss bootstrap (95% CI): RF − DC = −0.076 [−0.174, +0.006]; RF − Market =
+0.039 [−0.046, +0.121]; DC − Market = +0.115 [−0.007, +0.246]. Nothing reaches
significance on 64 matches, but the direction is striking: **2022 was so upset-rich
(Saudi Arabia over Argentina; Japan over Germany and Spain; Morocco's semifinal) that
the flattest forecaster, Elo, posted the best log-loss — nominally ahead of the market
itself.** The sharp Dixon-Coles paid the largest penalty for its confidence.

**Variable importance (top of 28 features).** t_attack 0.149, o_defense 0.122,
o_attack 0.113, t_defense 0.077, o_rank_pos 0.075, o_gdp_pc 0.066 — the four ability
parameters occupy four of the top five slots. **The original's central mechanism claim
replicates.**

**Table 5.3 — pre-tournament Monte-Carlo simulation (20,000 runs): P(champion),
P(reach final), P(reach semifinal).**

| | Dixon-Coles | | | Hybrid RF | | |
|---|---|---|---|---|---|---|
| team | champ | final | SF | champ | final | SF |
| Brazil | **0.418** | 0.511 | 0.708 | 0.057 | 0.116 | 0.232 |
| Argentina | 0.244 | 0.343 | 0.624 | 0.058 | 0.118 | 0.233 |
| Uruguay | 0.079 | 0.208 | 0.338 | 0.060 | 0.122 | 0.238 |
| France | 0.013 | 0.048 | 0.121 | **0.103** | 0.175 | 0.299 |
| Netherlands | — | — | — | 0.100 | 0.173 | 0.296 |
| Belgium | — | — | — | 0.082 | 0.142 | 0.247 |

Most likely finals: DC — Brazil v Uruguay (9.2%), Argentina v Uruguay (5.5%),
Argentina v Brazil (4.9%); RF — France v Netherlands (1.9%), Belgium v Netherlands
(1.4%). **What happened: Argentina beat France on penalties.** Both models missed the
exact final; the assessment differs sharply in style. Dixon-Coles was over-concentrated
(Brazil 41.8% exited in the quarterfinals) yet had the actual champion second at 24.4%
and the runner-up nowhere; the hybrid RF was so flat (maximum 10.3%) that it contained
the actual finalists near the top (France 1st, Argentina 9th at 5.8%) while saying
little. The actual champion's probability: DC 24.4%, RF 5.8% — a clear point for the
ranking model, the mirror image of their match-level scores that year.

### 5.2 The 2026 World Cup (prospective, through the Round of 16)

Hybrid RF trained on all seven tournaments 1998–2022; frozen at 11 June 2026.
Evaluation: the 96 matches played by 8 July 2026 (72 group, 24 knockout).

**Table 5.4 — match-level evaluation, 96 matches.**

| Method | Class. rate ↑ | Likelihood ↑ | RPS ↓ | Log-loss ↓ |
|---|---|---|---|---|
| Hybrid RF | 0.656 | 0.4683 | 0.1570 | 0.8559 |
| Dixon-Coles | 0.667 | 0.4877 | 0.1466 | 0.8152 |
| Elo | 0.635 | 0.4629 | 0.1683 | 0.9395 |
| **Market (de-vigged consensus)** | **0.698** | **0.5168** | **0.1405** | **0.7906** |

Paired log-loss bootstrap (95% CI): RF − DC = +0.041 [−0.001, +0.081] (just short of
significance); RF − Market = **+0.065 [+0.015, +0.114] (significant)**; DC − Market =
+0.025 [−0.024, +0.073]. In this more chalk-friendly edition the ordering inverts
relative to 2022: the market is best on every criterion, **the hybrid RF is
significantly worse than the market**, and plain Dixon-Coles is statistically
indistinguishable from it.

**Variable importance.** t_attack 0.140, o_defense 0.118, o_attack 0.115, o_rank_pos
0.081, t_defense 0.072 — abilities dominate again; the mechanism claim replicates in
both experiments.

**Table 5.6 — pre-tournament simulation (20,000 runs), top 8 by P(champion).**

| | Dixon-Coles | Hybrid RF |
|---|---|---|
| Argentina | **0.161** | 0.042 |
| Spain | 0.122 | 0.078 |
| England | 0.080 | 0.100 |
| Portugal | 0.061 | 0.075 |
| France | 0.061 | **0.115** |
| Brazil | 0.059 | 0.047 |
| Germany | 0.047 | 0.081 |
| Morocco | 0.042 | — |

Most likely finals pre-tournament: DC — Argentina v Spain (3.6%); RF — England v
France (2.0%). Reality check to date: Norway — outside both models' top ten — has
eliminated Brazil and reached the quarterfinals.

**Decomposing the market's edge (is it modeling or information?).** The models are
frozen at 11 June by design; the market's closing line updates continuously. Splitting
the 96 matches by segment separates the two explanations
(`scripts/market_edge_decomposition.py`):

| Segment (log-loss) | Market | DC frozen | DC updated* | edge (Mkt − DC frozen) |
|---|---|---|---|---|
| Matchday 1 (1–24) | 1.0051 | **1.0009** | 1.0045 | −0.004 (model ahead) |
| Matchday 2 (25–48) | **0.7406** | 0.7455 | 0.7531 | +0.005 |
| Matchday 3 (49–72) | **0.6973** | 0.7623 | 0.7672 | +0.065 |
| Knockout (73–96) | **0.7196** | 0.7519 | 0.7473 | +0.032 |

\* Dixon-Coles refit before every match date, i.e. updating on tournament *scores*.

Three observations. First, **on matchday 1 — when the market possesses no tournament
information the model lacks — the frozen model is level with (nominally ahead of) the
closing line.** The market's edge appears with the tournament and peaks exactly on
matchday 3, the rotation round, where qualified teams rest starters and the market
sees team sheets an hour before kickoff. Second, updating the model on tournament
*results* does not help (DC updated ≈ DC frozen): the market's advantage is *news* —
lineups, injuries, motivation — which no results-based model observes. Third, a minnow
split reverses the intuitive blind-spot story: on the 26 matches involving a team
ranked outside the top 60, **Dixon-Coles beats the market** (0.729 vs 0.767); the
market's entire edge sits on established-team matches. The correct reading of Table
5.4 is therefore not "the market out-models us" but "**the closing line out-informs
us, where there is information to have**."

**A news-aware ability model.** To test whether that diagnosis is *sufficient*, we
built the missing channel: a lineup/form adjustment to the Dixon-Coles intensities,
using only pre-kickoff information — the official starting XI (published ~1h before
kickoff, FIFA live API) and events of *earlier* matches. Two features per team-match,
both zero on matchday 1: **rotation** (1 − the share of the team's previous matches
today's starters started) and **form** (goal over-performance vs. the model so far,
weighted by whether today's XI contains the players who produced it — the
"player-is-having-a-great-tournament" term). Coefficients fit by Poisson ML on the
2018+2022 tournaments (no 2026 information): own rotation cuts own scoring
(β_own = +0.21), facing a rotated side boosts it strongly (β_opp = +1.23 — rotation
costs defensive cohesion more than attack), and form is real (γ = +0.23).
Out-of-sample on 2026:

| Segment (log-loss) | Market | DC frozen | DC+lineup | DC+lineup+form | RF frozen | RF+news |
|---|---|---|---|---|---|---|
| Matchday 2 | **0.7406** | 0.7455 | 0.7626 | 0.7848 | 0.7983 | 0.7900 |
| Matchday 3 | **0.6973** | 0.7623 | 0.7639 | 0.7937 | 0.8100 | 0.8334 |
| Knockout | 0.7196 | 0.7519 | 0.7239 | **0.7183** | 0.7833 | 0.7664 |
| All | **0.7906** | 0.8152 | 0.8128 | 0.8244 | 0.8559 | 0.8555 |

(The same layer applied to the hybrid RF uses coefficients calibrated on RF
intensities — β_own +0.12, β_opp +0.64, γ +0.15, smaller than the DC coefficients
because the forest's covariates already absorb part of what rotation signals.)

The result is sharply two-sided. **In the knockouts — once features rest on three-plus
matches of history — the news-adjusted Dixon-Coles reaches market level (0.7183 vs
0.7196, nominally ahead), closing what was a +0.032 gap.** Early in the tournament the
same features are one-or-two-match noise and *hurt*; and the matchday-3 gap does
**not** close — the market's rotation-round edge evidently reflects more than team
sheets (dead-rubber motivation, tactical intent). Net across all 96 matches the layer
is a wash for both bases (DC: +0.009 [−0.045, +0.073]; RF: −0.001 [−0.029, +0.030] vs
their frozen versions), and the news-adjusted RF remains significantly behind the
market (+0.065 [+0.009, +0.117]) — the layer narrows its knockout gap (0.783 → 0.766)
but cannot rescue the weaker base model. The information story is therefore confirmed
but refined: lineups and form are *part* of the market's mid-tournament information
advantage — enough for the ability model to match it by the knockout rounds — while
the matchday-3 residue requires information no lineup sheet carries. Note this is the
paper's only T2-vs-T2⁺ comparison (§2.3): the news models see lineups and prior
events, exactly the class of information the closing line has — though the line still
holds strictly more.

**Table 5.7 — conditional champion probabilities given the reached quarterfinal
bracket** (France–Morocco, Spain–Belgium → SF1; Norway–England,
Argentina–Switzerland → SF2; exact bracket computation; models still frozen at
11 June).

| Team | Hybrid RF | Dixon-Coles | Elo | Market (outrights, 8 Jul) |
|---|---|---|---|---|
| Argentina | 0.098 | **0.275** | **0.293** | 0.186 |
| France | **0.220** | 0.114 | 0.158 | **0.323** |
| Spain | 0.145 | 0.207 | 0.217 | 0.198 |
| England | 0.204 | 0.143 | 0.129 | 0.155 |
| Belgium | 0.129 | 0.063 | 0.067 | 0.027 |
| Morocco | 0.056 | 0.088 | 0.063 | 0.027 |
| Norway | 0.063 | 0.063 | 0.031 | 0.055 |
| Switzerland | 0.085 | 0.047 | 0.042 | 0.027 |

The rating models nominate **Argentina**; the hybrid RF sides with the market on
**France**. Information-set note (§2.3): the model columns are T0 forecasts
conditioned only on the realized bracket, while the outright odds are T2⁺ — the
market's France 32% partly *is* France's tournament form, which the frozen models
cannot see by design. This disagreement resolves within two weeks and is recorded
here in advance.

## 6. Concluding remarks

Across 160 fully out-of-sample World Cup matches (2022 complete, 2026 through the
Round of 16), four findings:

1. **The hybrid's advantage over its parents does not clearly replicate.** In the
   leave-one-tournament-out comparison the hybrid RF and the plain ability model are
   statistically tied (both clearly beat Elo); in the two prospective experiments the
   hybrid never significantly beats Dixon-Coles, loses to it nominally in 2026, and is
   the only model significantly behind the market there. The forest and covariates add
   robustness in some editions (2014–2022 LOO; upset-heavy 2022, where its flatness
   helped at match level) but no reliable edge over the abilities it is built on.
2. **The original's mechanism claim replicates robustly.** In every fit, the four
   ability parameters dominate variable importance, exactly as Groll et al. report —
   which is precisely why the wrapper struggles to beat the ability model alone,
   particularly with the squad-structure covariates unavailable (§2.1).
3. **Tournament character dominates method ranking.** Chaotic 2022 rewarded flat
   forecasts (Elo nominally beat even the market on log-loss); chalk-friendly 2026
   rewards sharp ones (Dixon-Coles statistically ties the market). No method dominates
   across both, and 64–96 match samples leave most pairwise comparisons inside
   confidence intervals. Single-tournament evaluations — including the original's 2018
   showcase and this paper's tables — should be read with that humility.
4. **The market remains the benchmark to beat — but its edge is information, not
   modeling — and part of it is reproducible.** Pooled across both tournaments the
   consensus odds are the best-calibrated forecaster, yet the §5.2 decomposition shows
   the frozen ability model is level with the closing line whenever the two share the
   same information set (matchday 1; minnow matches). Feeding the model the same
   pre-kickoff news the market has — official lineups and within-tournament player
   form, coefficients calibrated on 2018+2022 — recovers **market-level accuracy in
   the knockout rounds** (0.7183 vs 0.7196), while the matchday-3 rotation-round
   residue survives: that information (motivation, tactical intent in dead rubbers)
   is not on the team sheet.

**Limitations.** Reduced covariate set (no market values / squad structure / coach
variables); extra-time scores in the response for knockouts; penalties as a coin flip;
2026 conclusions provisional until the final on 19 July 2026 — re-running
`paper_eval.py` after each round updates every table.

---

### Reproducibility

```bash
python scripts/scrape_wc2022_odds.py     # market odds 2022 (BetExplorer)
python scripts/scrape_wc2026_odds.py     # market odds 2026 (OddsPortal; needs Chrome)
python paper_eval.py                     # regenerates paper/results.json (~5 min)
python scripts/calibrate_news_layer.py   # (β, γ) from 2018+2022 lineups/goals
python scripts/market_edge_decomposition.py   # §5.2 decomposition incl. news layer
```

Data provenance note: lineup/goal events come from FIFA's live-match API for all three
tournaments. Transfermarkt player values were considered for value-weighted rotation
but rejected: season pages serve *post*-tournament values (e.g. E. Fernández €80m on
the "2022" page — his post-WC price), which would leak outcomes into calibration, and
Wayback coverage of era-correct pages is too sparse. Equal-weighted rotation avoids
the leak; value-weighting is future work.

### References

- Dixon, M.J. & Coles, S.G. (1997). Modelling association football scores and
  inefficiencies in the football betting market. *JRSS C*, 46(2).
- Groll, A., Ley, C., Schauberger, G. & Van Eetvelde, H. (2019). A hybrid random
  forest to predict soccer matches in international tournaments. *JQAS*, 15(4).
- Ley, C., Van de Wiele, T. & Van Eetvelde, H. (2019). Ranking soccer teams on the
  basis of their current strength. *Statistical Modelling*, 19(1).
- Maher, M.J. (1982). Modelling association football scores. *Statistica
  Neerlandica*, 36(3).
