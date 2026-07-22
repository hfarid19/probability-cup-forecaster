"""Model abstraction: produce calibrated match outcome probabilities."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import Outcome


class Model(ABC):
    """Abstract base for any model that predicts match outcome probabilities.

    Concrete subclasses (Elo, Dixon-Coles, the hybrid random forest) implement
    :meth:`match_probabilities` however they like; callers depend only on this
    common interface.
    """

    @abstractmethod
    def match_probabilities(self, home: str, away: str) -> dict[Outcome, float]:
        """Predict the probability of each outcome for a single match.

        Args:
            home: Name of the home team.
            away: Name of the away team.

        Returns:
            dict[Outcome, float]: Probabilities keyed by HOME/DRAW/AWAY that sum to ~1.0.
        """
        ...
