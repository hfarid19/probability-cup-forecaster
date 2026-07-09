"""Model abstraction: produce calibrated match outcome probabilities."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import Outcome


class Model(ABC):
    @abstractmethod
    def match_probabilities(self, home: str, away: str) -> dict[Outcome, float]:
        """Return P(HOME/DRAW/AWAY) for the match; values must sum to ~1.0."""
        ...
