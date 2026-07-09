"""Walk-forward Elo backtest (Milestone 3).

Method: step through matches in chronological order. For each match we FIRST predict
(using only ratings learned from prior matches → genuinely out-of-sample), record the
prediction, THEN update ratings on the result. Evaluation can be restricted to a recent
window and/or specific tournaments while still training on the full history.

The gate: the model must beat the base-rate baseline on log-loss. (Beating the *market*
closing line is the stronger test and needs historical odds — a later milestone.)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..model.elo import EloModel
from ..types import Outcome
from .metrics import (
    ReliabilityBin,
    accuracy,
    base_rate_baseline,
    brier_score,
    log_loss,
    reliability_curve,
)

CLASSES = ("HOME", "DRAW", "AWAY")


def match_outcome(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "HOME"
    if home_score < away_score:
        return "AWAY"
    return "DRAW"


@dataclass
class BacktestReport:
    params: dict
    n_eval: int
    eval_start: str | None
    tournaments: tuple[str, ...] | None
    log_loss: float
    brier: float
    accuracy: float
    baseline_log_loss: float
    reliability: list[ReliabilityBin] = field(default_factory=list)

    @property
    def skill_vs_baseline(self) -> float:
        """Fractional log-loss improvement over the base-rate predictor (>0 = skill)."""
        if self.baseline_log_loss == 0:
            return 0.0
        return (self.baseline_log_loss - self.log_loss) / self.baseline_log_loss


def run_elo_backtest(
    df: pd.DataFrame,
    *,
    eval_start: str | None = None,
    tournaments: tuple[str, ...] | None = None,
    home_advantage: float = 60.0,
    draw_factor: float = 0.28,
    k: float = 20.0,
    n_bins: int = 10,
) -> BacktestReport:
    model = EloModel(home_advantage=home_advantage, draw_factor=draw_factor, k=k)
    eval_start_ts = pd.Timestamp(eval_start) if eval_start else None
    tset = set(tournaments) if tournaments else None

    probs: list[dict] = []
    outcomes: list[str] = []

    for r in df.itertuples(index=False):
        in_eval = eval_start_ts is None or r.date >= eval_start_ts
        if tset is not None:
            in_eval = in_eval and r.tournament in tset
        if in_eval:
            p = model.match_probabilities(r.home_team, r.away_team, bool(r.neutral))
            probs.append({c: p[Outcome(c)] for c in CLASSES})
            outcomes.append(match_outcome(r.home_score, r.away_score))
        # Train on every match (even those we evaluate, but only AFTER predicting them).
        model.update(r.home_team, r.away_team, r.home_score, r.away_score, bool(r.neutral))

    if not outcomes:
        raise ValueError("No matches in the evaluation window/tournament filter.")

    base = base_rate_baseline(outcomes, CLASSES)
    baseline_ll = log_loss([base] * len(outcomes), outcomes)

    return BacktestReport(
        params={"home_advantage": home_advantage, "draw_factor": draw_factor, "k": k},
        n_eval=len(outcomes),
        eval_start=eval_start,
        tournaments=tournaments,
        log_loss=log_loss(probs, outcomes),
        brier=brier_score(probs, outcomes, CLASSES),
        accuracy=accuracy(probs, outcomes),
        baseline_log_loss=baseline_ll,
        reliability=reliability_curve(probs, outcomes, CLASSES, n_bins=n_bins),
    )


def tune_elo(
    df: pd.DataFrame,
    *,
    eval_start: str | None = None,
    tournaments: tuple[str, ...] | None = None,
    home_advantages: tuple[float, ...] = (40, 60, 80, 100),
    draw_factors: tuple[float, ...] = (0.22, 0.26, 0.30, 0.34),
    ks: tuple[float, ...] = (15, 20, 30, 40),
) -> tuple[BacktestReport, list[BacktestReport]]:
    """Small grid search minimizing out-of-sample log-loss. Returns (best, all)."""
    all_reports: list[BacktestReport] = []
    for ha in home_advantages:
        for dfac in draw_factors:
            for k in ks:
                all_reports.append(run_elo_backtest(
                    df, eval_start=eval_start, tournaments=tournaments,
                    home_advantage=ha, draw_factor=dfac, k=k,
                ))
    best = min(all_reports, key=lambda r: r.log_loss)
    return best, all_reports
