"""Bookmaker odds -> fair probabilities -> implied goal expectations.

Sharp closing prices are the hardest benchmark to beat (the replication paper's
§5.2 finding: the market's edge is information, not modelling). So for any question
with a direct betting line we anchor on the de-vigged price, and for the derived
("exotic") questions we back out the goal-expectation pair (lambda_home, lambda_away)
that reproduces the book's 1X2 + totals, then compute everything analytically.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from wc_trader.model.dixon_coles import DixonColesModel


def devig_1x2(odds_home: float, odds_draw: float, odds_away: float) -> dict[str, float]:
    """Decimal odds -> proportionally de-vigged {HOME, DRAW, AWAY} probabilities.

    Matches wc_trader.experiment.load_market_probs so market rows are comparable.
    """
    inv = np.array([1 / odds_home, 1 / odds_draw, 1 / odds_away])
    inv /= inv.sum()
    return {"HOME": float(inv[0]), "DRAW": float(inv[1]), "AWAY": float(inv[2])}


def american_to_prob(american: int) -> float:
    """A single American-odds line -> its raw implied probability (vig included)."""
    return 100 / (american + 100) if american > 0 else -american / (-american + 100)


def devig_two_way(prob_yes_raw: float, prob_no_raw: float) -> float:
    """De-vig a two-way (yes/no) market to a fair P(yes)."""
    return prob_yes_raw / (prob_yes_raw + prob_no_raw)


def score_grid(lam_home: float, lam_away: float, rho: float, max_goals: int = 10) -> np.ndarray:
    """Dixon-Coles corrected P(home=x, away=y) grid for a given lambda pair."""
    from scipy.stats import poisson

    gv = np.arange(max_goals + 1)
    g = np.outer(poisson.pmf(gv, lam_home), poisson.pmf(gv, lam_away))
    g[0, 0] *= 1 - lam_home * lam_away * rho
    g[0, 1] *= 1 + lam_home * rho
    g[1, 0] *= 1 + lam_away * rho
    g[1, 1] *= 1 - rho
    g = np.clip(g, 0.0, None)
    return g / g.sum()


def _grid_summary(g: np.ndarray) -> dict[str, float]:
    total = np.add.outer(np.arange(g.shape[0]), np.arange(g.shape[1]))
    return {
        "home": float(np.tril(g, -1).sum()),
        "draw": float(np.trace(g)),
        "under25": float(g[total <= 2].sum()),
    }


def implied_lambdas(
    devigged: dict[str, float],
    prob_under_2_5: float,
    rho: float,
    x0: tuple[float, float] = (1.4, 1.1),
) -> tuple[float, float]:
    """Back out (lambda_home, lambda_away) reproducing the book's 1X2 + O/U 2.5.

    The book prices *outcomes*; we need the *rates* behind them to derive questions
    the book doesn't quote (halftime states, timing windows, same-goal halves, ...).
    """
    target = {"home": devigged["HOME"], "draw": devigged["DRAW"], "under25": prob_under_2_5}

    def loss(logp: np.ndarray) -> float:
        s = _grid_summary(score_grid(np.exp(logp[0]), np.exp(logp[1]), rho))
        return sum((s[k] - target[k]) ** 2 for k in target)

    res = minimize(loss, np.log(x0), method="Nelder-Mead")
    return float(np.exp(res.x[0])), float(np.exp(res.x[1]))


def blended_lambdas(
    dc: DixonColesModel,
    home: str,
    away: str,
    market_lams: tuple[float, float],
    weight_market: float = 0.5,
) -> tuple[float, float]:
    """Blend model lambdas with market-implied lambdas.

    The frozen model underrates in-form teams (paper §5.2); the market prices that
    information but overreacts to narrative. A blend keeps the model's structure
    while borrowing the market's information. Default 50/50.
    """
    lam_dc = float(np.exp(dc.attack.get(home, 0.0) - dc.defense.get(away, 0.0)))
    mu_dc = float(np.exp(dc.attack.get(away, 0.0) - dc.defense.get(home, 0.0)))
    w = weight_market
    return (w * market_lams[0] + (1 - w) * lam_dc, w * market_lams[1] + (1 - w) * mu_dc)


# --- binary-question probabilities (was markets.py) ---

# Empirical shares measured from wc_trader.data.lineups goal events (2022 + 2026
# World Cups, regulation only). Recompute with base_rates.py after each round.
FIRST_HALF_GOAL_SHARE = 0.42
LATE_WINDOW_GOAL_SHARE = 0.29  # minute >= ~76 ("after 2nd hydration break")


def prob_result(lams: tuple[float, float], rho: float) -> dict[str, float]:
    g = score_grid(*lams, rho)
    return {
        "home_win": float(np.tril(g, -1).sum()),
        "draw": float(np.trace(g)),
        "away_win": float(np.triu(g, 1).sum()),
    }


def prob_total_goals_at_most(lams: tuple[float, float], rho: float, n: int) -> float:
    g = score_grid(*lams, rho)
    total = np.add.outer(np.arange(g.shape[0]), np.arange(g.shape[1]))
    return float(g[total <= n].sum())


def prob_both_teams_score(lams: tuple[float, float], rho: float) -> float:
    g = score_grid(*lams, rho)
    return float(g[1:, 1:].sum())


def prob_tied_at_half(lams: tuple[float, float], rho: float) -> float:
    fh = FIRST_HALF_GOAL_SHARE
    g1 = score_grid(lams[0] * fh, lams[1] * fh, rho * 0.5)
    return float(np.trace(g1))


def prob_same_goals_each_half(lams: tuple[float, float], rho: float) -> float:
    """P(#goals in H1 == #goals in H2) — a question crowds rarely compute."""
    fh = FIRST_HALF_GOAL_SHARE
    g1 = score_grid(lams[0] * fh, lams[1] * fh, rho * 0.5)
    g2 = score_grid(lams[0] * (1 - fh), lams[1] * (1 - fh), 0.0)

    def total_pmf(g: np.ndarray, kmax: int = 8) -> np.ndarray:
        t = np.add.outer(np.arange(g.shape[0]), np.arange(g.shape[1]))
        return np.array([g[t == k].sum() for k in range(kmax + 1)])

    p1, p2 = total_pmf(g1), total_pmf(g2)
    return float((p1 * p2).sum())


def prob_goal_in_window(lams: tuple[float, float], share_of_goals: float) -> float:
    """P(>=1 goal in a window holding `share_of_goals` of expected scoring)."""
    rate = (lams[0] + lams[1]) * share_of_goals
    return float(1 - np.exp(-rate))


def prob_late_goal(lams: tuple[float, float]) -> float:
    return prob_goal_in_window(lams, LATE_WINDOW_GOAL_SHARE)


def prob_from_poisson_count(expected: float, threshold: int) -> float:
    """Generic P(count >= threshold) for a Poisson stat (shots, corners, SOT)."""
    from scipy.stats import poisson

    return float(1 - poisson.cdf(threshold - 1, expected))


def apply_boldness(fair: float, crowd: float, factor: float = 1.5,
                   floor: float | None = None, cap: float | None = None) -> int:
    """Amplify a fair-value disagreement with the crowd for relative-Brier upside.

    Pushes `fair` further from `crowd` by `factor`, then clamps to [1, 99] and any
    caller floor/cap (e.g. never fade a penalty-taking golden-boot leader below a
    floor; keep card markets above a floor). Returns an integer 1..99 (Cup format).
    """
    bold = crowd + (fair - crowd) * factor
    if floor is not None:
        bold = max(bold, floor)
    if cap is not None:
        bold = min(bold, cap)
    return int(round(min(99, max(1, bold))))
