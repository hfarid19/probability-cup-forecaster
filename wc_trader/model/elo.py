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
    def __init__(
        self,
        ratings: dict[str, float] | None = None,
        home_advantage: float = 60.0,
        draw_factor: float = 0.28,
        k: float = 20.0,
    ):
        self.ratings: dict[str, float] = dict(ratings or {})
        self.home_advantage = home_advantage
        self.draw_factor = draw_factor   # peak draw probability when teams are evenly matched
        self.k = k                       # update step size

    def rating(self, team: str) -> float:
        return self.ratings.setdefault(team, 1500.0)

    def _expected_home(self, home: str, away: str, neutral: bool = False) -> float:
        # No home advantage at neutral venues (most World Cup games).
        adv = 0.0 if neutral else self.home_advantage
        diff = (self.rating(home) + adv) - self.rating(away)
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))

    def match_probabilities(self, home: str, away: str, neutral: bool = False) -> dict[Outcome, float]:
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
        """Online Elo update after a result (margin-agnostic baseline)."""
        exp = self._expected_home(home, away, neutral)
        score = 1.0 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0.0
        delta = self.k * (score - exp)
        self.ratings[home] = self.rating(home) + delta
        self.ratings[away] = self.rating(away) - delta
