"""Elo baseline model.

Simple, robust, hard to overfit — the BENCHMARK every fancier model (Dixon-Coles) must
beat on out-of-sample log-loss. Ships with a working rating/update rule plus a heuristic
draw model so it's usable immediately for the skeleton.
"""
from __future__ import annotations

import math

from ..types import Outcome
from .base import Model


class EloModel(Model):
    """Elo rating baseline that turns two teams' ratings into outcome probabilities.

    Each team carries a single rating; the home team gets a fixed bonus at
    non-neutral venues. A logistic function maps the rating gap to an expected
    home result, and a Gaussian bump splits off draw probability. Ratings can be
    updated online after each result.
    """

    def __init__(
        self,
        ratings: dict[str, float] | None = None,
        home_advantage: float = 60.0,
        draw_factor: float = 0.28,
        k: float = 20.0,
    ):
        """Initialize the model with optional starting ratings and tuning constants.

        Args:
            ratings: Initial team -> rating map, copied so the caller's dict is untouched.
                Unseen teams default to 1500 on first access.
            home_advantage: Rating points added to the home team at non-neutral venues.
            draw_factor: Peak draw probability when the two teams are evenly matched.
            k: Update step size controlling how fast ratings move after a result.
        """
        self.ratings: dict[str, float] = dict(ratings or {})
        self.home_advantage = home_advantage
        self.draw_factor = draw_factor   # peak draw probability when teams are evenly matched
        self.k = k                       # update step size

    def rating(self, team: str) -> float:
        """Return a team's current rating, seeding unseen teams at 1500.

        Args:
            team: Team name to look up.

        Returns:
            float: The team's rating (1500 if it had no prior rating).
        """
        return self.ratings.setdefault(team, 1500.0)

    def _expected_home(self, home: str, away: str, neutral: bool = False) -> float:
        """Compute the logistic expected home result from the rating gap.

        The value lies in (0, 1) and behaves like an expected points share
        (1 = certain home win, 0.5 = even, 0 = certain away win).

        Args:
            home: Home team name.
            away: Away team name.
            neutral: If True, no home advantage is applied (most World Cup games).

        Returns:
            float: Expected home score share in (0, 1).
        """
        # No home advantage at neutral venues (most World Cup games).
        adv = 0.0 if neutral else self.home_advantage
        diff = (self.rating(home) + adv) - self.rating(away)
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))

    def match_probabilities(self, home: str, away: str, neutral: bool = False) -> dict[Outcome, float]:
        """Predict HOME/DRAW/AWAY probabilities for a single match.

        The expected home share sets the win/loss split; a Gaussian draw bump
        (largest for evenly matched teams) is carved out, then all three are
        normalized to sum to 1.

        Args:
            home: Home team name.
            away: Away team name.
            neutral: If True, no home advantage is applied.

        Returns:
            dict[Outcome, float]: Normalized probabilities keyed by HOME/DRAW/AWAY.
        """
        p_home_exp = self._expected_home(home, away, neutral)
        # Draw mass peaks when the expected score is ~0.5 (evenly matched).
        p_draw = self.draw_factor * math.exp(-((p_home_exp - 0.5) ** 2) / 0.08)
        p_home = p_home_exp * (1 - p_draw)
        p_away = (1 - p_home_exp) * (1 - p_draw)
        total = p_home + p_draw + p_away
        return {
            Outcome.HOME: p_home / total,
            Outcome.DRAW: p_draw / total,
            Outcome.AWAY: p_away / total,
        }

    def update(self, home: str, away: str, home_goals: int, away_goals: int,
               neutral: bool = False) -> None:
        """Apply an online Elo update after one result (margin-agnostic baseline).

        The home team gains and the away team loses the same amount, proportional
        to the surprise between the actual result (win/draw/loss) and the expected
        home share.

        Args:
            home: Home team name.
            away: Away team name.
            home_goals: Goals scored by the home team.
            away_goals: Goals scored by the away team.
            neutral: If True, no home advantage is used in the expectation.
        """
        exp = self._expected_home(home, away, neutral)
        score = 1.0 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0.0
        delta = self.k * (score - exp)
        self.ratings[home] = self.rating(home) + delta
        self.ratings[away] = self.rating(away) - delta
