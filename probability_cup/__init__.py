"""Live-forecasting layer that applied the replication model to the Jump Trading
Probability Cup (2026 World Cup knockout rounds).

The Cup posed binary questions per match ("Will the match have <=2 goals?",
"Will Messi score?", ...) scored by Brier loss *relative to the crowd average*.
Being right isn't enough; you score by being better calibrated than the field,
which rewards fading the crowd's systematic biases (over-goals, over-cards,
star-scorer hype) rather than chasing them.

Pipeline:
    forecast.py  de-vig bookmaker prices -> fair probabilities; back out the
                 goal-expectation (lambda) pair a match implies; derive each binary
                 question's probability from a Dixon-Coles score grid blended with
                 market lambdas, plus tournament base rates and a boldness helper.
    base_rates.py  empirical rates for the no-line questions, from goal-event data.
    client.py    thin SportsPredict REST client (key from env), batch submit + patch.

Nothing here hardcodes a key or an entrant identity; see README "Probability Cup".
"""
