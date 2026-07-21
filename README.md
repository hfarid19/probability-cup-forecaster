# wc-trader

A football (soccer) match prediction model, rebuilt from a published research paper (Groll,
Ley, Schauberger & Van Eetvelde, 2019, "A hybrid random forest to predict soccer
matches in international tournaments"), tested on the 2022 and 2026 World Cups, and
then used to compete in a live forecasting competition (the Jump Trading Probability
Cup).

It includes three prediction models, from simple to advanced:

- **Elo**: the classic chess rating system, adapted for football.
- **Dixon-Coles**: a statistical model that gives each team an attack and a defense
  strength and turns those into goal probabilities.
- **Hybrid random forest**: the paper's model. A machine-learning method that mixes
  team information (rankings, economy, host status) with the Dixon-Coles strength ratings.

Around them sits a scoring system that measures how accurate the predictions are, and a
"freeze" rule that makes sure every prediction only uses information available before the
match was played, so nothing is predicted with hindsight.

## What the project found

- The advanced machine-learning model clearly beats the simple baselines (Elo and plain
  guessing), but it never clearly beats the simpler statistical model it is built on top
  of. Tested carefully, the two are a tie. The reason: the model leans almost entirely on
  the team-strength ratings, so the extra machine learning adds little.
- The betting market's advantage comes from information, not from a cleverer model. Once
  the model is fed the actual lineups and recent player form, it matches the bookmakers'
  accuracy in the knockout rounds (0.718 vs 0.720 on the scoring metric, where lower is
  better).
- Put to the test: used as an automated bot in the Jump Trading Probability Cup, this
  model finished in the top 3% of 3,867 forecasters (details below).

## How the Dixon-Coles model works

Dixon-Coles is the heart of the project. Both the paper's main model and the competition
bot are built on it, so it is worth explaining in plain terms.

The idea is simple. Every team gets two numbers:

- an **attack** rating: how many goals it tends to score, and
- a **defense** rating: how many goals it tends to let in.

To predict a match, the model combines the home team's attack with the away team's defense
to estimate how many goals the home team is likely to score, then does the same the other
way around for the away team. That gives an expected goal count for each side, for example
1.8 for one team and 0.9 for the other.

It then treats goals as random events and uses a standard counting formula (the Poisson
distribution, the usual way to model how many times something happens) to turn those
expected goals into the probability of every exact scoreline: 0-0, 1-0, 2-1, and so on.
Adding up the relevant scorelines gives any answer you want, such as the chance of a home
win, a draw, more or fewer than a set number of goals, or both teams scoring.

The "Dixon-Coles" part is a small fix on top of that. Plain goal-counting slightly
misjudges very low-scoring results (0-0, 1-0, 0-1, 1-1), because in real football those
tight games happen a bit more often than pure chance would predict. Dixon and Coles added
a correction for exactly those four scorelines, which is what makes the model handle the
draw-heavy nature of football well.

The two ratings for each team are learned from past results, with recent matches counting
more than old ones, since teams change over time. Nobody assigns the ratings; the model
works them out from match history.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python paper_eval.py   # re-run the full analysis, writes paper/results.json (about 5 min)
pytest                 # run the tests (no internet needed)
```

All data is public and cached under `data/raw/`: international match results, FIFA
rankings, World Bank economic figures, and match calendars, lineups, and betting odds.

## Layout

```
wc_trader/
  experiment.py       # prepares each tournament's data using only pre-match information
  model/              # the three prediction models (Elo, Dixon-Coles, random forest)
  data/               # loads the raw inputs (results, rankings, economy, lineups)
  backtest/           # scores predictions and simulates whole tournaments
probability_cup/      # uses the model to compete in the live forecasting contest
paper_eval.py         # runs the whole analysis and saves the results
paper/                # results.json (the saved output of paper_eval.py)
```

## The Probability Cup (the live competition)

`probability_cup/` puts the model to work in the Jump Trading Probability Cup, a live
forecasting contest over the 2026 World Cup knockout rounds. Each match had about 15
yes/no questions ("Will there be 2 goals or fewer?", "Will Messi score?", "Will there be
a goal late in the game?"). You are scored on how accurate your probabilities are compared
to the average of everyone else, so the goal is not just to be right, it is to be better
calibrated than the crowd. That rewards betting against the crowd's predictable habits,
like over-rating goals, cards, and star players.

The code reuses the same models from the research half:

- `forecast.py` turns bookmaker odds into fair probabilities. For questions the bookmakers
  do not cover, it works out how many goals each team is expected to score and runs those
  through the Dixon-Coles model to get an answer. It also has a helper that nudges a
  prediction further from the crowd when we genuinely disagree, for a competitive edge.
- `base_rates.py` measures how often things actually happen from real match data (for
  example, a substitute scores in 38% of matches and a late goal happens in 53%), for the
  questions that have no betting line.
- `client.py` is a small connector to the SportsPredict website that submits predictions.
  It reads the API key from an environment variable (`SPORTSPREDICT_KEY`), never from the
  code. Copy `.env.example` to `.env` and add your key.

```bash
export SPORTSPREDICT_KEY=sp_live_...   # your bot key, see .env.example
python -m probability_cup.base_rates   # refresh the "how often does X happen" numbers
pytest tests/test_probability_cup.py   # test the math (no internet or key needed)
```

Result: the bot finished in the top 3% of 3,867 forecasters, which fits the project's main
finding. The edge comes from good calibration and information, not from a more complicated
model.
