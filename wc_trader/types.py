"""Core domain types."""
from __future__ import annotations

from enum import Enum


class Outcome(str, Enum):
    """The three possible results of a football match from the home team's view.

    A ``str`` subclass so members compare equal to and serialize as their plain
    string value ("HOME"/"DRAW"/"AWAY").
    """

    HOME = "HOME"
    DRAW = "DRAW"
    AWAY = "AWAY"
