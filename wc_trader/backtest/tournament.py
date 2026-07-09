"""Monte-Carlo World Cup simulator, template-driven from FIFA's calendar API.

Mirrors §5 of Groll et al. (2019): sample every match score from a model, play out the
group stage (points → goal difference → goals scored; residual ties broken randomly as a
stand-in for head-to-head/lots), resolve the knockout template, and repeat n times to
estimate each team's probability of winning the title / reaching each stage.

The template is parsed from the cached FIFA calendar JSON (data/raw/fifa_calendar_<year>.json),
so both the 2022 (32-team) and 2026 (48-team, third-place slots like '3ABCDF') formats work
without hand-coded brackets. Third-place slot assignment — FIFA's combination table — is
solved per-simulation by backtracking over the slots' allowed-group sets.

Knockout draws after 90' advance by coin flip (extra time + penalties approximation).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ..data.fifa_rank import normalize_team

# Samples (home_goals, away_goals) for a neutral-venue match.
ScoreSampler = Callable[[str, str, np.random.RandomState], tuple[int, int]]


@dataclass
class KOMatch:
    number: int
    ref_a: str   # '1A' | '2B' | '3ABCDF' | 'W97' | 'RU101'
    ref_b: str


@dataclass
class TournamentSpec:
    groups: dict[str, list[str]]                 # 'A' -> 4 team names
    fixtures: list[tuple[str, str, str]]         # (group, home, away)
    ko: list[KOMatch]                            # sorted by match number
    final_number: int

    @property
    def teams(self) -> list[str]:
        return sorted(t for g in self.groups.values() for t in g)


def _team_name(side: dict | None) -> str | None:
    if not side:
        return None
    names = side.get("TeamName") or []
    # harmonize FIFA naming ("Korea Republic", "USA") to results-dataset names
    return normalize_team(names[0]["Description"]) if names else None


def parse_calendar(path: str | Path) -> TournamentSpec:
    data = json.loads(Path(path).read_text())
    groups: dict[str, list[str]] = {}
    fixtures: list[tuple[str, str, str]] = []
    ko: list[KOMatch] = []

    for m in data["Results"]:
        gname = (m.get("GroupName") or [{}])[0].get("Description")
        if gname:
            letter = gname.replace("Group", "").strip()
            h, a = _team_name(m.get("Home")), _team_name(m.get("Away"))
            if h and a:
                fixtures.append((letter, h, a))
                for t in (h, a):
                    if t not in groups.setdefault(letter, []):
                        groups[letter].append(t)
        else:
            ko.append(KOMatch(int(m["MatchNumber"]), m.get("PlaceHolderA") or "",
                              m.get("PlaceHolderB") or ""))

    ko.sort(key=lambda k: k.number)
    return TournamentSpec(groups=groups, fixtures=fixtures, ko=ko,
                          final_number=max(k.number for k in ko))


def _standings(order_key: dict[str, tuple]) -> list[str]:
    return sorted(order_key, key=lambda t: order_key[t], reverse=True)


def _assign_thirds(slots: list[tuple[int, str]], qualified: dict[str, str]) -> dict[int, str]:
    """Match third-place slots (match_number, allowed-groups string like '3ABCDF') to the
    qualifying groups by backtracking. Returns {match_number: team}."""
    slot_opts = []
    for num, ref in slots:
        allowed = [g for g in ref[1:] if g in qualified]
        slot_opts.append((num, allowed))
    slot_opts.sort(key=lambda s: len(s[1]))     # most constrained first

    assignment: dict[int, str] = {}

    def bt(i: int, used: frozenset) -> bool:
        if i == len(slot_opts):
            return True
        num, allowed = slot_opts[i]
        for g in allowed:
            if g not in used:
                assignment[num] = qualified[g]
                if bt(i + 1, used | {g}):
                    return True
                del assignment[num]
        return False

    if not bt(0, frozenset()):
        # No consistent assignment (shouldn't happen with FIFA's table); fall back greedily.
        used: set = set()
        for num, allowed in slot_opts:
            g = next((g for g in allowed if g not in used), None)
            if g is None:
                g = next(g for g in qualified if g not in used)
            used.add(g)
            assignment[num] = qualified[g]
    return assignment


def simulate_once(spec: TournamentSpec, sampler: ScoreSampler,
                  rng: np.random.RandomState) -> dict:
    # --- group stage ---
    pts: dict[str, float] = {}
    gd: dict[str, int] = {}
    gf: dict[str, int] = {}
    for _, h, a in spec.fixtures:
        hg, ag = sampler(h, a, rng)
        pts[h] = pts.get(h, 0) + (3 if hg > ag else 1 if hg == ag else 0)
        pts[a] = pts.get(a, 0) + (3 if ag > hg else 1 if hg == ag else 0)
        gd[h] = gd.get(h, 0) + hg - ag
        gd[a] = gd.get(a, 0) + ag - hg
        gf[h] = gf.get(h, 0) + hg
        gf[a] = gf.get(a, 0) + ag

    key = {t: (pts.get(t, 0), gd.get(t, 0), gf.get(t, 0), rng.random())
           for g in spec.groups.values() for t in g}

    pos: dict[str, str] = {}       # '1A' -> team
    thirds: dict[str, str] = {}    # group -> third-placed team
    for g, teams in spec.groups.items():
        table = _standings({t: key[t] for t in teams})
        for place, t in enumerate(table, start=1):
            pos[f"{place}{g}"] = t
        if len(table) >= 3:
            thirds[g] = table[2]

    # Best thirds (2026): rank across groups, keep as many as the template needs.
    third_slots = [(k.number, r) for k in spec.ko for r in (k.ref_a, k.ref_b)
                   if re.fullmatch(r"3[A-L]{2,}", r)]
    third_assignment: dict[tuple[int, str], str] = {}
    if third_slots:
        n_needed = len(third_slots)
        ranked = sorted(thirds, key=lambda g: key[thirds[g]], reverse=True)[:n_needed]
        qualified = {g: thirds[g] for g in ranked}
        by_num = _assign_thirds([(n, r) for n, r in third_slots], qualified)
        for n, r in third_slots:
            third_assignment[(n, r)] = by_num[n]

    # --- knockout ---
    winners: dict[int, str] = {}
    losers: dict[int, str] = {}
    participants: dict[int, tuple[str, str]] = {}

    def resolve(num: int, ref: str) -> str:
        if re.fullmatch(r"[12][A-L]", ref):
            return pos[ref]
        if re.fullmatch(r"3[A-L]{2,}", ref):
            return third_assignment[(num, ref)]
        if ref.startswith("W"):
            return winners[int(ref[1:])]
        if ref.startswith("RU"):
            return losers[int(ref[2:])]
        raise ValueError(f"unknown placeholder {ref!r}")

    for k in spec.ko:
        a, b = resolve(k.number, k.ref_a), resolve(k.number, k.ref_b)
        ag_, bg_ = sampler(a, b, rng)
        if ag_ == bg_:                                   # ET + pens ≈ coin flip
            w = a if rng.random() < 0.5 else b
        else:
            w = a if ag_ > bg_ else b
        winners[k.number] = w
        losers[k.number] = b if w == a else a
        participants[k.number] = (a, b)

    final = spec.final_number
    sf_nums = [int(r[1:]) for r in
               (next(k for k in spec.ko if k.number == final).ref_a,
                next(k for k in spec.ko if k.number == final).ref_b)]
    return {
        "champion": winners[final],
        "finalists": participants[final],
        "semifinalists": tuple(t for n in sf_nums for t in participants[n]),
    }


def simulate(spec: TournamentSpec, sampler: ScoreSampler, n_sims: int = 20000,
             seed: int = 42) -> tuple[dict[str, dict[str, float]], dict[tuple, float]]:
    """Returns ({team: {champion, final, semifinal}} probabilities,
                {(finalistA, finalistB): probability} for the most likely finals)."""
    rng = np.random.RandomState(seed)
    counts = {t: {"champion": 0, "final": 0, "semifinal": 0} for t in spec.teams}
    final_pairs: dict[tuple, int] = {}
    for _ in range(n_sims):
        out = simulate_once(spec, sampler, rng)
        counts[out["champion"]]["champion"] += 1
        for t in out["finalists"]:
            counts[t]["final"] += 1
        for t in out["semifinalists"]:
            counts[t]["semifinal"] += 1
        pair = tuple(sorted(out["finalists"]))
        final_pairs[pair] = final_pairs.get(pair, 0) + 1
    return ({t: {k: v / n_sims for k, v in c.items()} for t, c in counts.items()},
            {p: c / n_sims for p, c in sorted(final_pairs.items(), key=lambda x: -x[1])})


# ---- samplers ----

def dc_sampler(dc, cache: dict | None = None) -> ScoreSampler:
    """Sample scores from the Dixon-Coles corrected joint grid (neutral venue)."""
    grids: dict[tuple[str, str], np.ndarray] = cache if cache is not None else {}

    def sample(h: str, a: str, rng: np.random.RandomState) -> tuple[int, int]:
        g = grids.get((h, a))
        if g is None:
            g = dc.score_grid(h, a, neutral=True)
            grids[(h, a)] = g
        flat = rng.choice(g.size, p=g.ravel())
        return int(flat // g.shape[1]), int(flat % g.shape[1])

    return sample


def rf_sampler(lambdas: dict[tuple[str, str], tuple[float, float]]) -> ScoreSampler:
    """Sample independent Poisson scores from precomputed RF expected goals
    {(home, away): (lambda_home, lambda_away)} — the paper's §5 procedure."""
    def sample(h: str, a: str, rng: np.random.RandomState) -> tuple[int, int]:
        lh, la = lambdas[(h, a)]
        return int(rng.poisson(lh)), int(rng.poisson(la))

    return sample
