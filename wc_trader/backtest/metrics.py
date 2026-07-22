"""Scoring for probabilistic match forecasts.

Forecasts are dicts over the ordered outcome classes HOME/DRAW/AWAY; a model can be
accurate yet badly calibrated, so proper scoring rules (log-loss, Brier, RPS) matter
more than the classification rate.
"""
from __future__ import annotations

import math

EPS = 1e-15
CLASSES = ("HOME", "DRAW", "AWAY")


def match_outcome(home_score: int, away_score: int) -> str:
    """Classify a match result as HOME, DRAW, or AWAY from the final score.

    Args:
        home_score: Goals scored by the home team.
        away_score: Goals scored by the away team.

    Returns:
        str: "HOME", "AWAY", or "DRAW".
    """
    if home_score > away_score:
        return "HOME"
    if home_score < away_score:
        return "AWAY"
    return "DRAW"


def log_loss(probs: list[dict], outcomes: list[str]) -> float:
    """Compute the mean negative log-likelihood of the realized outcomes.

    Lower is better. Predicted probabilities are floored at EPS to keep the
    logarithm finite.

    Args:
        probs: Per-match forecasts, each a dict over the outcome classes.
        outcomes: Realized outcome class for each match.

    Returns:
        float: Mean negative log-likelihood.
    """
    return sum(-math.log(max(EPS, p[y])) for p, y in zip(probs, outcomes)) / len(outcomes)


def brier_score(probs: list[dict], outcomes: list[str],
                classes: tuple[str, ...] = CLASSES) -> float:
    """Compute the multiclass Brier score (mean squared error vs one-hot truth).

    Lower is better.

    Args:
        probs: Per-match forecasts, each a dict over the outcome classes.
        outcomes: Realized outcome class for each match.
        classes: The outcome classes to score over.

    Returns:
        float: Mean squared error against the one-hot true outcomes.
    """
    return sum(sum((p[c] - (1.0 if c == y else 0.0)) ** 2 for c in classes)
               for p, y in zip(probs, outcomes)) / len(outcomes)


def accuracy(probs: list[dict], outcomes: list[str]) -> float:
    """Compute the share of matches where the most likely class was the outcome.

    Args:
        probs: Per-match forecasts, each a dict over the outcome classes.
        outcomes: Realized outcome class for each match.

    Returns:
        float: Fraction of matches whose argmax-probability class was correct.
    """
    return sum(1 for p, y in zip(probs, outcomes) if max(p, key=p.get) == y) / len(outcomes)


def avg_likelihood(probs: list[dict], outcomes: list[str]) -> float:
    """Compute the mean predicted probability of the realized outcome.

    This is Groll et al.'s "likelihood" measure. Higher is better; roughly 1/3 is
    uninformed for three outcomes.

    Args:
        probs: Per-match forecasts, each a dict over the outcome classes.
        outcomes: Realized outcome class for each match.

    Returns:
        float: Mean probability assigned to the outcome that actually occurred.
    """
    return sum(p[y] for p, y in zip(probs, outcomes)) / len(outcomes)


def rps(probs: list[dict], outcomes: list[str],
        order: tuple[str, ...] = CLASSES) -> float:
    """Compute the mean ranked probability score over ordered outcomes.

    Outcomes are treated as ordered (home win > draw > away win). RPS is
    1/(K-1) * sum_k (cumulative_predicted_k - cumulative_observed_k)^2, which
    penalizes probability placed far from the true outcome. Lower is better.

    Args:
        probs: Per-match forecasts, each a dict over the outcome classes.
        outcomes: Realized outcome class for each match.
        order: The outcome classes in rank order.

    Returns:
        float: Mean ranked probability score.
    """
    k = len(order)
    total = 0.0
    for p, y in zip(probs, outcomes):
        cum_pred = cum_obs = 0.0
        s = 0.0
        for c in order[:-1]:
            cum_pred += p[c]
            cum_obs += 1.0 if c == y else 0.0   # stays 1 after the true outcome's cutpoint
            s += (cum_pred - cum_obs) ** 2
        total += s / (k - 1)
    return total / len(outcomes)


def paired_logloss_bootstrap(probs_a: list[dict], probs_b: list[dict], outcomes: list[str],
                             n_boot: int = 10000, seed: int = 42) -> dict:
    """Compare two models' per-match log-losses with a paired bootstrap.

    Works on the per-match log-loss difference (A minus B). Because both models
    score the same matches, match-difficulty variance cancels out. A negative mean
    difference means model A is better.

    Args:
        probs_a: Model A's per-match forecasts, each a dict over the classes.
        probs_b: Model B's per-match forecasts, each a dict over the classes.
        outcomes: Realized outcome class for each match.
        n_boot: Number of bootstrap resamples.
        seed: Seed for the resampling RNG.

    Returns:
        dict: {"mean_diff", "ci_low", "ci_high"} for the bootstrap 95% CI of the
            mean log-loss difference, plus "n" (the number of matches).
    """
    import random

    diffs = [-math.log(max(EPS, a[y])) + math.log(max(EPS, b[y]))
             for a, b, y in zip(probs_a, probs_b, outcomes)]
    n = len(diffs)
    rng = random.Random(seed)
    boot_means = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return {
        "mean_diff": sum(diffs) / n,
        "ci_low": boot_means[int(0.025 * n_boot)],
        "ci_high": boot_means[int(0.975 * n_boot)],
        "n": n,
    }
