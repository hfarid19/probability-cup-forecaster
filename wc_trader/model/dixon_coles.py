"""Dixon-Coles Poisson goals model (PRIMARY model — Milestone 4).

Each team has an ATTACK and DEFENSE strength; home goals ~ Poisson(λ), away ~ Poisson(μ):

    λ = exp(attack_home - defense_away + home_adv * is_home)
    μ = exp(attack_away - defense_home)

with the Dixon-Coles low-score correlation correction τ(x, y; λ, μ, ρ) that fixes the
0-0/1-1/1-0/0-1 probabilities Poisson alone gets wrong (the football draw structure).
Strengths are fit by weighted maximum likelihood (recent matches weighted more via a
half-life), then a Poisson score grid is summed into HOME/DRAW/AWAY probabilities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from ..types import Outcome
from .base import Model


def _tau(hg, ag, lam, mu, rho):
    """Compute the Dixon-Coles low-score correction factor for each match.

    Independent Poissons misprice the four low-score results (0-0, 0-1, 1-0,
    1-1); this returns the multiplicative correction that rescales those cells
    and leaves every other scoreline at 1.0. Vectorized over arrays.

    Args:
        hg: Array of home goals per match.
        ag: Array of away goals per match.
        lam: Array of home scoring rates (lambda) per match.
        mu: Array of away scoring rates (mu) per match.
        rho: Scalar low-score correlation parameter.

    Returns:
        np.ndarray: Correction factor per match, 1.0 outside the four low-score cells.
    """
    t = np.ones_like(lam, dtype=float)
    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)
    t[m00] = 1.0 - lam[m00] * mu[m00] * rho
    t[m01] = 1.0 + lam[m01] * rho
    t[m10] = 1.0 + mu[m10] * rho
    t[m11] = 1.0 - rho
    return t


class DixonColesModel(Model):
    """Dixon-Coles Poisson goals model fit by time-weighted maximum likelihood.

    Learns per-team attack and defense strengths plus a home-advantage term and
    the low-score correlation rho. Predictions build a Poisson score grid, apply
    the low-score correction, and sum it into HOME/DRAW/AWAY probabilities.
    """

    def __init__(self, max_goals: int = 10):
        """Initialize an unfitted model.

        Args:
            max_goals: Highest goal count per team included in the score grid.
        """
        self.attack: dict[str, float] = {}
        self.defense: dict[str, float] = {}
        self.home_adv: float = 0.0
        self.rho: float = 0.0
        self.max_goals = max_goals
        self._fitted = False

    def fit(self, df: pd.DataFrame, *, half_life_days: float | None = 540.0,
            ridge: float = 0.01, min_matches: int = 5) -> "DixonColesModel":
        """Estimate team strengths, home advantage, and rho by weighted MLE.

        Filters to teams with enough matches, weights each match by recency, then
        minimizes the ridge-penalized negative log-likelihood of the Dixon-Coles
        model. Results are stored on the instance.

        Args:
            df: Match table with home_team, away_team, home_score, away_score,
                neutral, and date columns.
            half_life_days: Recency half-life in days for match weights; None weights
                all matches equally.
            ridge: L2 penalty on attack and defense parameters for stability.
            min_matches: Minimum appearances for a team to be estimated.

        Returns:
            DixonColesModel: This instance, now fitted (for chaining).
        """
        # Keep teams with enough games for a stable estimate.
        counts = pd.concat([df["home_team"], df["away_team"]]).value_counts()
        teams = sorted(counts[counts >= min_matches].index)
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        d = df[df["home_team"].isin(idx) & df["away_team"].isin(idx)]
        h = d["home_team"].map(idx).to_numpy()
        a = d["away_team"].map(idx).to_numpy()
        hg = d["home_score"].to_numpy()
        ag = d["away_score"].to_numpy()
        not_neutral = (~d["neutral"].to_numpy().astype(bool)).astype(float)

        if half_life_days:
            age = (d["date"].max() - d["date"]).dt.days.to_numpy()
            w = 0.5 ** (age / half_life_days)
        else:
            w = np.ones(len(d))

        def neg_ll(params):
            atk = params[:n]
            dfn = params[n:2 * n]
            gamma, rho = params[2 * n], params[2 * n + 1]
            lam = np.exp(atk[h] - dfn[a] + gamma * not_neutral)
            mu = np.exp(atk[a] - dfn[h])
            tau = np.clip(_tau(hg, ag, lam, mu, rho), 1e-10, None)
            lp_home = hg * np.log(lam) - lam
            lp_away = ag * np.log(mu) - mu
            ll = w * (lp_home + lp_away + np.log(tau))
            return -ll.sum() + ridge * (atk @ atk + dfn @ dfn)

        x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25], [-0.05]])
        bounds = [(-3, 3)] * (2 * n) + [(-1, 1), (-0.2, 0.2)]
        res = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 200})

        p = res.x
        self.attack = {t: float(p[idx[t]]) for t in teams}
        self.defense = {t: float(p[n + idx[t]]) for t in teams}
        self.home_adv = float(p[2 * n])
        self.rho = float(p[2 * n + 1])
        self._fitted = True
        return self

    def score_grid(self, home: str, away: str, neutral: bool = False) -> np.ndarray:
        """Build the joint scoreline probability grid for a match.

        Entry ``grid[x, y]`` is P(home scores x, away scores y) from independent
        Poissons with the Dixon-Coles low-score correction applied and the whole
        grid renormalized to sum to 1. Unknown teams fall back to league-average
        strength.

        Args:
            home: Home team name.
            away: Away team name.
            neutral: If True, no home advantage is applied.

        Returns:
            np.ndarray: A (max_goals+1, max_goals+1) grid of scoreline probabilities.
        """
        # Unknown teams fall back to league-average strength (0).
        ah, dh = self.attack.get(home, 0.0), self.defense.get(home, 0.0)
        aa, da = self.attack.get(away, 0.0), self.defense.get(away, 0.0)
        adv = 0.0 if neutral else self.home_adv

        lam = float(np.exp(ah - da + adv))
        mu = float(np.exp(aa - dh))

        gv = np.arange(self.max_goals + 1)
        ph = poisson.pmf(gv, lam)   # P(home = x)
        pa = poisson.pmf(gv, mu)    # P(away = y)
        grid = np.outer(ph, pa)     # grid[x, y]

        # Apply the DC correction to the four low-score cells.
        grid[0, 0] *= 1.0 - lam * mu * self.rho
        grid[0, 1] *= 1.0 + lam * self.rho
        grid[1, 0] *= 1.0 + mu * self.rho
        grid[1, 1] *= 1.0 - self.rho
        grid = np.clip(grid, 0.0, None)
        return grid / grid.sum()

    def match_probabilities(self, home: str, away: str, neutral: bool = False) -> dict[Outcome, float]:
        """Predict HOME/DRAW/AWAY probabilities by summing the score grid.

        Sums the lower triangle (home wins), diagonal (draw), and upper triangle
        (away wins) of :meth:`score_grid`.

        Args:
            home: Home team name.
            away: Away team name.
            neutral: If True, no home advantage is applied.

        Returns:
            dict[Outcome, float]: Probabilities keyed by HOME/DRAW/AWAY summing to ~1.0.
        """
        grid = self.score_grid(home, away, neutral)
        return {
            Outcome.HOME: float(np.tril(grid, -1).sum()),   # x > y
            Outcome.DRAW: float(np.trace(grid)),            # x == y
            Outcome.AWAY: float(np.triu(grid, 1).sum()),    # x < y
        }
