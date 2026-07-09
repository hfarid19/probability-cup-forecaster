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
    if home_score > away_score:
        return "HOME"
    if home_score < away_score:
        return "AWAY"
    return "DRAW"


def log_loss(probs: list[dict], outcomes: list[str]) -> float:
    """Mean negative log-likelihood of the realized outcomes. Lower is better."""
    return sum(-math.log(max(EPS, p[y])) for p, y in zip(probs, outcomes)) / len(outcomes)


def brier_score(probs: list[dict], outcomes: list[str],
                classes: tuple[str, ...] = CLASSES) -> float:
    """Multiclass Brier score: mean squared error vs one-hot truth. Lower is better."""
    return sum(sum((p[c] - (1.0 if c == y else 0.0)) ** 2 for c in classes)
               for p, y in zip(probs, outcomes)) / len(outcomes)


def accuracy(probs: list[dict], outcomes: list[str]) -> float:
    """Share of matches where the argmax-probability class was the actual outcome."""
    return sum(1 for p, y in zip(probs, outcomes) if max(p, key=p.get) == y) / len(outcomes)


def avg_likelihood(probs: list[dict], outcomes: list[str]) -> float:
    """Mean predicted probability of the realized outcome (Groll et al.'s 'likelihood').

    Higher is better; 1/3 ≈ uninformed for three outcomes.
    """
    return sum(p[y] for p, y in zip(probs, outcomes)) / len(outcomes)


def rps(probs: list[dict], outcomes: list[str],
        order: tuple[str, ...] = CLASSES) -> float:
    """Mean ranked probability score over ORDERED outcomes (home win > draw > away win).

    RPS = 1/(K-1) * Σ_k (cumulative_predicted_k − cumulative_observed_k)².
    Penalizes probability placed far from the true outcome; lower is better.
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
    """Paired comparison of two models' per-match log-losses (A minus B).

    Returns mean difference and a bootstrap 95% CI. Negative mean = A better.
    Paired (same matches) so match-difficulty variance cancels out.
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
