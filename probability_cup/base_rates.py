"""Empirical base rates for exotic questions, from wc_trader goal-event data.

Half of the Cup's questions have no bookmaker line (same-goal halves, late-window
goals, sub involvement). For those the crowd guesses from intuition and misses
systematically; a measured base rate is the edge. Regulation only: base minute
<= 45 is first half, 46-90 second half, > 90 (extra time) and blank (shootout)
excluded. Re-run after each round to keep FIRST_HALF_GOAL_SHARE etc. current.
"""
from __future__ import annotations

import re

from wc_trader.data.lineups import load_lineups


def _base_minute(raw: str | None) -> int | None:
    """Parse a goal-minute string into its regulation base minute.

    Reads the leading integer of a minute label so stoppage-time forms collapse
    onto their base (45'+x -> 45, 90'+x -> 90). Extra-time goals (base > 90) are
    dropped by returning None.

    Args:
        raw: The raw minute label from a goal event, possibly None or blank.

    Returns:
        int | None: The base minute in [0, 90], or None if unparseable or beyond
            regulation.
    """
    m = re.match(r"(\d+)", str(raw or "").strip())
    if not m:
        return None
    base = int(m.group(1))
    return None if base > 90 else base  # 45'+x -> 45, 90'+x -> 90, ET dropped


def _match_features(rec: dict) -> dict:
    """Extract per-match goal-timing and substitute features from a lineup record.

    Walks both sides' goal events (regulation only) and derives goal counts by
    half, late-window goals, and whether any goal involved a substitute (as
    scorer or assister).

    Args:
        rec: A lineup record with "home" and "away" sub-dicts, each holding
            "players" and "goals" lists.

    Returns:
        dict: Feature counts and flags keyed by "n_goals", "h1", "h2", "late",
            and "sub_involved".
    """
    starters = {
        str(p["id"]): p.get("starter", True)
        for side in ("home", "away")
        for p in rec[side]["players"]
    }
    goals = []
    for side in ("home", "away"):
        for gl in rec[side]["goals"]:
            bm = _base_minute(gl.get("minute"))
            if bm is not None:
                asst = str(gl["assist"]) if gl.get("assist") else None
                goals.append((bm, str(gl.get("player")), asst))
    h1 = sum(1 for bm, *_ in goals if bm <= 45)
    return {
        "n_goals": len(goals),
        "h1": h1,
        "h2": len(goals) - h1,
        "late": sum(1 for bm, *_ in goals if bm >= 76),
        "sub_involved": any(
            starters.get(sc) is False or (a and starters.get(a) is False)
            for _, sc, a in goals
        ),
    }


def compute(years: tuple[int, ...] = (2022, 2026)) -> dict[str, float]:
    """Aggregate base rates across the given World Cups (played matches only).

    Loads lineups for each year, keeps matches that were actually played (a
    non-empty home roster), and averages the per-match features into the
    tournament base rates the exotic-question forecasts rely on.

    Args:
        years: World Cup years to pool together.

    Returns:
        dict[str, float]: Base-rate metrics keyed by "n_matches",
            "goals_per_match", "first_half_goal_share", "p_sub_involved",
            "p_late_goal", "p_same_goals_each_half", and "p_at_most_2_goals".
    """
    feats = [
        _match_features(r)
        for y in years
        for r in load_lineups(y)
        if r["home"]["players"]
    ]
    n = len(feats)
    total_goals = sum(f["n_goals"] for f in feats)
    return {
        "n_matches": n,
        "goals_per_match": total_goals / n,
        "first_half_goal_share": sum(f["h1"] for f in feats) / max(1, total_goals),
        "p_sub_involved": sum(f["sub_involved"] for f in feats) / n,
        "p_late_goal": sum(f["late"] > 0 for f in feats) / n,
        "p_same_goals_each_half": sum(f["h1"] == f["h2"] for f in feats) / n,
        "p_at_most_2_goals": sum(f["n_goals"] <= 2 for f in feats) / n,
    }


if __name__ == "__main__":
    for k, v in compute().items():
        print(f"{k:28s} {v:.3f}" if isinstance(v, float) else f"{k:28s} {v}")
