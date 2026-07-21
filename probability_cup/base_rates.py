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
    m = re.match(r"(\d+)", str(raw or "").strip())
    if not m:
        return None
    base = int(m.group(1))
    return None if base > 90 else base  # 45'+x -> 45, 90'+x -> 90, ET dropped


def _match_features(rec: dict) -> dict:
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
    """Aggregate base rates across the given World Cups (played matches only)."""
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
