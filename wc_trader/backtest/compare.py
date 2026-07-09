"""Elo vs Dixon-Coles comparison on internationals (Milestone 4).

Walk forward through matches. Elo updates online (out-of-sample per match). Dixon-Coles
is refit periodically (every `refit_days`) on a trailing window with half-life weighting,
then held until the next refit. Both models are scored on the SAME eval matches — only
those where both teams are known to the current DC fit — so the log-loss comparison is
apples-to-apples. DC earns its place only if it beats Elo here.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..model.dixon_coles import DixonColesModel
from ..model.elo import EloModel
from ..types import Outcome
from .metrics import accuracy, base_rate_baseline, log_loss
from .simulator import CLASSES, match_outcome


@dataclass
class CompareReport:
    n_eval: int
    eval_start: str
    elo_log_loss: float
    dc_log_loss: float
    baseline_log_loss: float
    elo_accuracy: float
    dc_accuracy: float

    @property
    def winner(self) -> str:
        return "Dixon-Coles" if self.dc_log_loss < self.elo_log_loss else "Elo"


def compare_models(
    df: pd.DataFrame,
    *,
    eval_start: str,
    refit_days: int = 365,
    train_window_years: int = 8,
    half_life_days: float = 540.0,
    min_matches: int = 8,
) -> CompareReport:
    eval_ts = pd.Timestamp(eval_start)
    elo = EloModel()
    dc: DixonColesModel | None = None
    last_fit: pd.Timestamp | None = None

    elo_probs: list[dict] = []
    dc_probs: list[dict] = []
    outcomes: list[str] = []

    for r in df.itertuples(index=False):
        if r.date >= eval_ts:
            # Refit Dixon-Coles when stale, on a trailing window of prior matches only.
            if dc is None or (r.date - last_fit).days >= refit_days:
                lo = r.date - pd.DateOffset(years=train_window_years)
                train = df[(df.date >= lo) & (df.date < r.date)]
                dc = DixonColesModel().fit(train, half_life_days=half_life_days, min_matches=min_matches)
                last_fit = r.date

            # Only score matches both models can genuinely predict (both teams in DC fit).
            if r.home_team in dc.attack and r.away_team in dc.attack:
                ep = elo.match_probabilities(r.home_team, r.away_team, bool(r.neutral))
                dp = dc.match_probabilities(r.home_team, r.away_team, bool(r.neutral))
                elo_probs.append({c: ep[Outcome(c)] for c in CLASSES})
                dc_probs.append({c: dp[Outcome(c)] for c in CLASSES})
                outcomes.append(match_outcome(r.home_score, r.away_score))

        elo.update(r.home_team, r.away_team, r.home_score, r.away_score, bool(r.neutral))

    if not outcomes:
        raise ValueError("No eval matches scored — widen the window or lower min_matches.")

    base = base_rate_baseline(outcomes, CLASSES)
    return CompareReport(
        n_eval=len(outcomes),
        eval_start=eval_start,
        elo_log_loss=log_loss(elo_probs, outcomes),
        dc_log_loss=log_loss(dc_probs, outcomes),
        baseline_log_loss=log_loss([base] * len(outcomes), outcomes),
        elo_accuracy=accuracy(elo_probs, outcomes),
        dc_accuracy=accuracy(dc_probs, outcomes),
    )
