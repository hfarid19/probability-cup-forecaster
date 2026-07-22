"""Monte-Carlo World Cup simulator, template-driven from FIFA's calendar API.

Mirrors §5 of Groll et al. (2019): sample every match score from a model, play out the
group stage (points → goal difference → goals scored; residual ties broken randomly as a
stand-in for head-to-head/lots), resolve the knockout template, and repeat n times to
estimate each team's probability of winning the title / reaching each stage.

The template is parsed from the cached FIFA calendar JSON (data/raw/fifa_calendar_<year>.json),
so both the 2022 (32-team) and 2026 (48-team, third-place slots like '3ABCDF') formats work
without hand-coded brackets. Third-place slot assignment (FIFA's combination table) is
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
    """A single knockout fixture from the calendar template.

    Args:
        number: The FIFA match number (knockout matches are ordered by it).
        ref_a: Placeholder for one side, e.g. '1A', '2B', '3ABCDF', 'W97', 'RU101'.
        ref_b: Placeholder for the other side, in the same notation.
    """
    number: int
    ref_a: str   # '1A' | '2B' | '3ABCDF' | 'W97' | 'RU101'
    ref_b: str


@dataclass
class TournamentSpec:
    """Parsed tournament template: groups, group fixtures, and knockout bracket.

    Args:
        groups: Group letter to its list of team names, e.g. 'A' -> 4 teams.
        fixtures: Group-stage matches as (group, home, away) tuples.
        ko: Knockout fixtures sorted by match number.
        final_number: Match number of the final.
    """
    groups: dict[str, list[str]]                 # 'A' -> 4 team names
    fixtures: list[tuple[str, str, str]]         # (group, home, away)
    ko: list[KOMatch]                            # sorted by match number
    final_number: int

    @property
    def teams(self) -> list[str]:
        """Return every team in the tournament, sorted by name.

        Returns:
            list[str]: All team names across all groups, sorted.
        """
        return sorted(t for g in self.groups.values() for t in g)


def _team_name(side: dict | None) -> str | None:
    """Extract and normalize a team name from a calendar match side.

    Args:
        side: The "Home" or "Away" object from the calendar JSON, or None.

    Returns:
        str | None: The results-dataset team name, or None if the side has no team.
    """
    if not side:
        return None
    names = side.get("TeamName") or []
    # harmonize FIFA naming ("Korea Republic", "USA") to results-dataset names
    return normalize_team(names[0]["Description"]) if names else None


def parse_calendar(path: str | Path) -> TournamentSpec:
    """Parse a cached FIFA calendar JSON file into a TournamentSpec.

    Group matches (those carrying a group name) become fixtures and populate the
    group rosters; the remaining matches become the knockout template.

    Args:
        path: Path to the cached FIFA calendar JSON file.

    Returns:
        TournamentSpec: The parsed groups, fixtures, and knockout bracket.
    """
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
    """Rank teams from best to worst by a comparable sort key.

    Args:
        order_key: Team name to its ranking tuple (higher sorts higher), e.g.
            (points, goal difference, goals for, tie-break random).

    Returns:
        list[str]: Team names ordered best to worst.
    """
    return sorted(order_key, key=lambda t: order_key[t], reverse=True)


def _assign_thirds(slots: list[tuple[int, str]], qualified: dict[str, str]) -> dict[int, str]:
    """Assign qualifying third-placed teams to their knockout slots by backtracking.

    Each slot allows a fixed set of groups (encoded in its reference string, e.g.
    '3ABCDF'). Slots are filled most-constrained-first; if no consistent assignment
    exists (which should not happen with FIFA's table), a greedy fallback is used.

    Args:
        slots: (match_number, allowed-groups string) for each third-place slot.
        qualified: Group letter to the third-placed team that qualified from it.

    Returns:
        dict[int, str]: Match number to the team assigned to that slot.
    """
    slot_opts = []
    for num, ref in slots:
        allowed = [g for g in ref[1:] if g in qualified]
        slot_opts.append((num, allowed))
    slot_opts.sort(key=lambda s: len(s[1]))     # most constrained first

    assignment: dict[int, str] = {}

    def bt(i: int, used: frozenset) -> bool:
        """Recursively try to fill slot i onwards; return True on success.

        Args:
            i: Index of the slot to fill next.
            used: Group letters already consumed by earlier slots.

        Returns:
            bool: True if a consistent assignment for slots i.. exists.
        """
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
    """Play out one full tournament and report who reached the late stages.

    Simulates every group fixture, ranks the groups (points, then goal difference,
    then goals for, with a random final tie-break), assigns any best-third slots,
    then resolves the knockout bracket. Knockout draws after 90 minutes are decided
    by a coin flip (extra time plus penalties approximation).

    Args:
        spec: The parsed tournament template.
        sampler: Callback that samples (home_goals, away_goals) for a fixture.
        rng: Random state driving score sampling and tie-breaks.

    Returns:
        dict: {"champion": team, "finalists": (a, b), "semifinalists": (..4 teams)}.
    """
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
        """Resolve a knockout placeholder reference to a concrete team name.

        Args:
            num: Match number of the fixture using this reference.
            ref: Placeholder such as '1A', '3ABCDF', 'W97', or 'RU101'.

        Returns:
            str: The team currently filling that placeholder.
        """
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
    final_match = next(k for k in spec.ko if k.number == final)
    sf_nums = [int(r[1:]) for r in (final_match.ref_a, final_match.ref_b)]
    return {
        "champion": winners[final],
        "finalists": participants[final],
        "semifinalists": tuple(t for n in sf_nums for t in participants[n]),
    }


def simulate(spec: TournamentSpec, sampler: ScoreSampler, n_sims: int = 20000,
             seed: int = 42) -> tuple[dict[str, dict[str, float]], dict[tuple, float]]:
    """Estimate stage-reaching probabilities by Monte-Carlo over many tournaments.

    Runs simulate_once n_sims times and turns the counts into probabilities.

    Args:
        spec: The parsed tournament template.
        sampler: Callback that samples (home_goals, away_goals) for a fixture.
        n_sims: Number of tournament simulations to run.
        seed: Seed for the simulation RNG.

    Returns:
        tuple: A pair of dicts. The first maps each team to its
            {champion, final, semifinal} probabilities. The second maps each
            finalist pair (finalistA, finalistB) to its probability, ordered from
            most to least likely.
    """
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
    """Build a score sampler backed by the Dixon-Coles joint score grid.

    Scores are drawn from the corrected joint distribution at a neutral venue. Each
    matchup's grid is computed once and cached so repeated simulations are cheap.

    Args:
        dc: A fitted Dixon-Coles model exposing score_grid(home, away, neutral).
        cache: Optional dict reused as the (home, away) -> grid cache.

    Returns:
        ScoreSampler: A callable that samples (home_goals, away_goals).
    """
    grids: dict[tuple[str, str], np.ndarray] = cache if cache is not None else {}

    def sample(h: str, a: str, rng: np.random.RandomState) -> tuple[int, int]:
        """Sample one match score for home team h against away team a.

        Args:
            h: Home team name.
            a: Away team name.
            rng: Random state used to draw from the score grid.

        Returns:
            tuple[int, int]: Sampled (home_goals, away_goals).
        """
        g = grids.get((h, a))
        if g is None:
            g = dc.score_grid(h, a, neutral=True)
            grids[(h, a)] = g
        flat = rng.choice(g.size, p=g.ravel())
        return int(flat // g.shape[1]), int(flat % g.shape[1])

    return sample


def rf_sampler(lambdas: dict[tuple[str, str], tuple[float, float]]) -> ScoreSampler:
    """Build a score sampler that draws independent Poisson goals per team.

    Uses precomputed random-forest expected goals (the paper's section 5 procedure).

    Args:
        lambdas: (home, away) to (lambda_home, lambda_away) expected-goals rates.

    Returns:
        ScoreSampler: A callable that samples (home_goals, away_goals).
    """
    def sample(h: str, a: str, rng: np.random.RandomState) -> tuple[int, int]:
        """Sample one match score for home team h against away team a.

        Args:
            h: Home team name.
            a: Away team name.
            rng: Random state used to draw the Poisson goal counts.

        Returns:
            tuple[int, int]: Sampled (home_goals, away_goals).
        """
        lh, la = lambdas[(h, a)]
        return int(rng.poisson(lh)), int(rng.poisson(la))

    return sample
