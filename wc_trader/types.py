"""Core domain types."""
from __future__ import annotations

from enum import Enum


class Outcome(str, Enum):
    HOME = "HOME"
    DRAW = "DRAW"
    AWAY = "AWAY"
