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
from scipy.stats import poisson

from wc_trader.model.dixon_coles import DixonColesModel


def devig_1x2(odds_home: float, odds_draw: float, odds_away: float) -> dict[str, float]:
    """Convert decimal 1X2 odds into proportionally de-vigged probabilities.

    Strips the bookmaker margin by normalising the inverse odds so they sum to one.
    Matches wc_trader.experiment.load_market_probs so market rows are comparable.

    Args:
        odds_home: Decimal odds for the home win.
        odds_draw: Decimal odds for the draw.
        odds_away: Decimal odds for the away win.

    Returns:
        dict[str, float]: Fair probabilities keyed by "HOME", "DRAW", "AWAY".
    """
    inv = np.array([1 / odds_home, 1 / odds_draw, 1 / odds_away])
    inv /= inv.sum()
    return {"HOME": float(inv[0]), "DRAW": float(inv[1]), "AWAY": float(inv[2])}


def american_to_prob(american: int) -> float:
    """Convert a single American-odds line to its raw implied probability.

    The result still includes the bookmaker vig; de-vig separately if a fair
    probability is needed.

    Args:
        american: American (moneyline) odds, positive for underdogs and negative
            for favourites.

    Returns:
        float: The raw implied probability in [0, 1].
    """
    return 100 / (american + 100) if american > 0 else -american / (-american + 100)


def devig_two_way(prob_yes_raw: float, prob_no_raw: float) -> float:
    """De-vig a two-way (yes/no) market to a fair P(yes).

    Args:
        prob_yes_raw: Raw implied probability of the "yes" outcome.
        prob_no_raw: Raw implied probability of the "no" outcome.

    Returns:
        float: The fair probability of "yes" after removing the margin.
    """
    return prob_yes_raw / (prob_yes_raw + prob_no_raw)


def score_grid(lam_home: float, lam_away: float, rho: float, max_goals: int = 10) -> np.ndarray:
    """Build the Dixon-Coles corrected scoreline probability grid for a lambda pair.

    Takes independent Poisson scoring rates for each side, applies the Dixon-Coles
    low-score correction, then normalises so the grid sums to one.

    Args:
        lam_home: Expected goals for the home side.
        lam_away: Expected goals for the away side.
        rho: Dixon-Coles low-score dependence parameter.
        max_goals: Highest goal count per side to include in the grid.

    Returns:
        np.ndarray: A (max_goals+1, max_goals+1) array where entry [x, y] is
            P(home scores x, away scores y).
    """
    gv = np.arange(max_goals + 1)
    g = np.outer(poisson.pmf(gv, lam_home), poisson.pmf(gv, lam_away))
    g[0, 0] *= 1 - lam_home * lam_away * rho
    g[0, 1] *= 1 + lam_home * rho
    g[1, 0] *= 1 + lam_away * rho
    g[1, 1] *= 1 - rho
    g = np.clip(g, 0.0, None)
    return g / g.sum()


def _total_goals_matrix(g: np.ndarray) -> np.ndarray:
    """Matrix T where T[x, y] = x + y, the combined goal total for each scoreline.

    Args:
        g: A scoreline probability grid from score_grid.

    Returns:
        np.ndarray: An integer matrix of total goals matching the shape of g.
    """
    return np.add.outer(np.arange(g.shape[0]), np.arange(g.shape[1]))


def _grid_summary(g: np.ndarray) -> dict[str, float]:
    """Summarise a scoreline grid into the outcomes the lambda solver targets.

    Args:
        g: A scoreline probability grid from score_grid.

    Returns:
        dict[str, float]: P(home win), P(draw), and P(<=2 total goals) keyed by
            "home", "draw", and "under25".
    """
    total = _total_goals_matrix(g)
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
    Fits the lambda pair by minimising squared error against the target outcomes.

    Args:
        devigged: Fair 1X2 probabilities keyed by "HOME", "DRAW", "AWAY".
        prob_under_2_5: Fair probability of under 2.5 total goals.
        rho: Dixon-Coles low-score dependence parameter.
        x0: Initial (lambda_home, lambda_away) guess for the optimiser.

    Returns:
        tuple[float, float]: The fitted (lambda_home, lambda_away) pair.
    """
    target = {"home": devigged["HOME"], "draw": devigged["DRAW"], "under25": prob_under_2_5}

    def loss(logp: np.ndarray) -> float:
        """Squared error between the grid summary and the target outcomes."""
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

    Args:
        dc: Fitted Dixon-Coles model providing attack/defense strengths.
        home: Home team name (key into the model's strength tables).
        away: Away team name (key into the model's strength tables).
        market_lams: The market-implied (lambda_home, lambda_away) pair.
        weight_market: Weight on the market lambdas; the model gets 1 - weight.

    Returns:
        tuple[float, float]: The blended (lambda_home, lambda_away) pair.
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
    """Compute 1X2 outcome probabilities from a lambda pair.

    Args:
        lams: The (lambda_home, lambda_away) expected-goals pair.
        rho: Dixon-Coles low-score dependence parameter.

    Returns:
        dict[str, float]: Probabilities keyed by "home_win", "draw", "away_win".
    """
    g = score_grid(*lams, rho)
    return {
        "home_win": float(np.tril(g, -1).sum()),
        "draw": float(np.trace(g)),
        "away_win": float(np.triu(g, 1).sum()),
    }


def prob_total_goals_at_most(lams: tuple[float, float], rho: float, n: int) -> float:
    """Compute P(total goals <= n) from a lambda pair.

    Args:
        lams: The (lambda_home, lambda_away) expected-goals pair.
        rho: Dixon-Coles low-score dependence parameter.
        n: Inclusive upper bound on total match goals.

    Returns:
        float: The probability that the combined goal total is at most n.
    """
    g = score_grid(*lams, rho)
    total = _total_goals_matrix(g)
    return float(g[total <= n].sum())


def prob_both_teams_score(lams: tuple[float, float], rho: float) -> float:
    """Compute P(both teams score at least one goal) from a lambda pair.

    Args:
        lams: The (lambda_home, lambda_away) expected-goals pair.
        rho: Dixon-Coles low-score dependence parameter.

    Returns:
        float: The probability that both sides score.
    """
    g = score_grid(*lams, rho)
    return float(g[1:, 1:].sum())


def prob_tied_at_half(lams: tuple[float, float], rho: float) -> float:
    """Compute P(scores level at half time) from a full-match lambda pair.

    Scales each side's expected goals by the first-half share and halves the
    dependence parameter to approximate a single-half score grid.

    Args:
        lams: The (lambda_home, lambda_away) full-match expected-goals pair.
        rho: Dixon-Coles low-score dependence parameter for the full match.

    Returns:
        float: The probability the match is level at half time.
    """
    fh = FIRST_HALF_GOAL_SHARE
    g1 = score_grid(lams[0] * fh, lams[1] * fh, rho * 0.5)
    return float(np.trace(g1))


def prob_same_goals_each_half(lams: tuple[float, float], rho: float) -> float:
    """Compute P(goals in H1 == goals in H2), a question crowds rarely compute.

    Splits the match into two independent half grids (scaled by the first-half
    share) and sums the probability that both halves produce the same goal total.

    Args:
        lams: The (lambda_home, lambda_away) full-match expected-goals pair.
        rho: Dixon-Coles low-score dependence parameter for the full match.

    Returns:
        float: The probability both halves have equal goal totals.
    """
    fh = FIRST_HALF_GOAL_SHARE
    g1 = score_grid(lams[0] * fh, lams[1] * fh, rho * 0.5)
    g2 = score_grid(lams[0] * (1 - fh), lams[1] * (1 - fh), 0.0)

    def total_pmf(g: np.ndarray, kmax: int = 8) -> np.ndarray:
        """PMF of total goals in a half grid, for totals 0..kmax."""
        t = _total_goals_matrix(g)
        return np.array([g[t == k].sum() for k in range(kmax + 1)])

    p1, p2 = total_pmf(g1), total_pmf(g2)
    return float((p1 * p2).sum())


def prob_goal_in_window(lams: tuple[float, float], share_of_goals: float) -> float:
    """Compute P(>=1 goal in a window holding a given share of expected scoring).

    Models goals in the window as Poisson with rate equal to the total expected
    goals times the window's share.

    Args:
        lams: The (lambda_home, lambda_away) expected-goals pair.
        share_of_goals: Fraction of total expected goals falling in the window.

    Returns:
        float: The probability of at least one goal in the window.
    """
    rate = (lams[0] + lams[1]) * share_of_goals
    return float(1 - np.exp(-rate))


def prob_late_goal(lams: tuple[float, float]) -> float:
    """Compute P(>=1 goal in the late window) using the measured late-goal share.

    Args:
        lams: The (lambda_home, lambda_away) expected-goals pair.

    Returns:
        float: The probability of at least one goal in the late window.
    """
    return prob_goal_in_window(lams, LATE_WINDOW_GOAL_SHARE)


def prob_from_poisson_count(expected: float, threshold: int) -> float:
    """Compute P(count >= threshold) for a Poisson-distributed match stat.

    Useful for count markets such as shots, corners, or shots on target.

    Args:
        expected: Poisson mean (expected count) of the statistic.
        threshold: Inclusive lower bound on the count.

    Returns:
        float: The probability the count is at least threshold.
    """
    return float(1 - poisson.cdf(threshold - 1, expected))


def apply_boldness(fair: float, crowd: float, factor: float = 1.5,
                   floor: float | None = None, cap: float | None = None) -> int:
    """Amplify a fair-value disagreement with the crowd for relative-Brier upside.

    Pushes `fair` further from `crowd` by `factor`, then clamps to [1, 99] and any
    caller floor/cap (e.g. never fade a penalty-taking golden-boot leader below a
    floor; keep card markets above a floor).

    Args:
        fair: Our fair probability (0..100 scale) for the question.
        crowd: The crowd-average probability (0..100 scale) to fade.
        factor: Multiplier on the gap between fair and crowd.
        floor: Optional lower clamp applied before the [1, 99] bound.
        cap: Optional upper clamp applied before the [1, 99] bound.

    Returns:
        int: The bold prediction as an integer 1..99 (Cup format).
    """
    bold = crowd + (fair - crowd) * factor
    if floor is not None:
        bold = max(bold, floor)
    if cap is not None:
        bold = min(bold, cap)
    return int(round(min(99, max(1, bold))))
