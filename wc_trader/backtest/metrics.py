"""Calibration & accuracy metrics for probabilistic match predictions.

A model can be accurate yet badly calibrated (its 70%s don't happen 70% of the time),
and miscalibration is what bleeds money in betting. So we score with BOTH proper scoring
rules (log-loss, Brier) and a reliability curve.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

EPS = 1e-15


def log_loss(probs: list[dict], outcomes: list[str]) -> float:
    """Mean negative log-likelihood of the realized outcomes. Lower is better."""
    total = 0.0
    for p, y in zip(probs, outcomes):
        total += -math.log(max(EPS, p[y]))
    return total / len(outcomes)


def brier_score(probs: list[dict], outcomes: list[str], classes: tuple[str, ...]) -> float:
    """Multiclass Brier score: mean squared error vs one-hot truth. Lower is better."""
    total = 0.0
    for p, y in zip(probs, outcomes):
        total += sum((p[c] - (1.0 if c == y else 0.0)) ** 2 for c in classes)
    return total / len(outcomes)


def accuracy(probs: list[dict], outcomes: list[str]) -> float:
    """Share of matches where the argmax-probability class was the actual outcome."""
    hits = sum(1 for p, y in zip(probs, outcomes) if max(p, key=p.get) == y)
    return hits / len(outcomes)


@dataclass
class ReliabilityBin:
    lo: float
    hi: float
    n: int
    mean_predicted: float
    observed_freq: float


def reliability_curve(probs: list[dict], outcomes: list[str], classes: tuple[str, ...],
                      n_bins: int = 10) -> list[ReliabilityBin]:
    """Pool all (predicted_prob, hit) pairs across classes into probability bins.

    A well-calibrated model has mean_predicted ≈ observed_freq in every bin.
    """
    edges = [i / n_bins for i in range(n_bins + 1)]
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, y in zip(probs, outcomes):
        for c in classes:
            pr = p[c]
            idx = min(n_bins - 1, int(pr * n_bins))
            buckets[idx].append((pr, 1 if c == y else 0))

    out: list[ReliabilityBin] = []
    for i, bucket in enumerate(buckets):
        if not bucket:
            continue
        preds = [pr for pr, _ in bucket]
        hits = [h for _, h in bucket]
        out.append(ReliabilityBin(
            lo=edges[i], hi=edges[i + 1], n=len(bucket),
            mean_predicted=sum(preds) / len(preds),
            observed_freq=sum(hits) / len(hits),
        ))
    return out


def avg_likelihood(probs: list[dict], outcomes: list[str]) -> float:
    """Mean predicted probability of the realized outcome (Groll et al.'s 'likelihood').

    Higher is better; 1/3 ≈ uninformed for three outcomes.
    """
    return sum(p[y] for p, y in zip(probs, outcomes)) / len(outcomes)


def rps(probs: list[dict], outcomes: list[str],
        order: tuple[str, ...] = ("HOME", "DRAW", "AWAY")) -> float:
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


def base_rate_baseline(outcomes: list[str], classes: tuple[str, ...]) -> dict:
    """The trivial predictor: always predict the overall frequency of each class.

    Any real model MUST beat this on log-loss to claim it has skill.
    """
    n = len(outcomes)
    return {c: outcomes.count(c) / n for c in classes}
