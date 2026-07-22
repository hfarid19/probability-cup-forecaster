"""Exact 8-team knockout bracket solver.

Given four QF pairs (bracket order: winners of pairs 0,1 meet in SF1; winners of 2,3 in
SF2) and a `p_beat(a, b)` callback returning P(a eliminates b in a knockout tie), compute
each team's exact probability of winning the QF, reaching the final, and lifting the
trophy. With 8 teams the probability tree is tiny, so no Monte-Carlo is needed.

Knockout advance probability from a 3-way (win/draw/loss in 90') model:
    P(advance) = P(win90) + 0.5 * P(draw90)
i.e. extra time and penalties treated as a coin flip (the standard simplification).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ChampionForecast:
    """Per-team probabilities for the three stages of an 8-team knockout.

    Each dict maps a team name to a probability. The three stages are cumulative
    (winning the final implies reaching it, which implies winning the quarterfinal).

    Args:
        p_qf: Probability each team wins its quarterfinal.
        p_final: Probability each team reaches the final.
        p_champion: Probability each team wins the tournament.
    """
    p_qf: dict[str, float] = field(default_factory=dict)      # win quarterfinal
    p_final: dict[str, float] = field(default_factory=dict)   # reach the final
    p_champion: dict[str, float] = field(default_factory=dict)


def solve_bracket(pairs: list[tuple[str, str]],
                  p_beat: Callable[[str, str], float]) -> ChampionForecast:
    """Compute exact stage probabilities for an 8-team knockout bracket.

    Walks the (small) probability tree analytically: quarterfinals, then the two
    semifinals (pairs 0 vs 1 and 2 vs 3), then the final. No Monte-Carlo is needed
    because there are only 8 teams.

    Args:
        pairs: The four quarterfinal pairs in bracket order.
        p_beat: Callback returning P(a eliminates b) for a single knockout tie.

    Returns:
        ChampionForecast: Per-team probabilities of winning the quarterfinal,
            reaching the final, and winning the tournament.
    """
    if len(pairs) != 4:
        raise ValueError("expected exactly 4 quarterfinal pairs in bracket order")

    fc = ChampionForecast()

    # QF: distribution over each pair's winner
    qf_win: list[dict[str, float]] = []
    for a, b in pairs:
        pa = p_beat(a, b)
        qf_win.append({a: pa, b: 1.0 - pa})
        fc.p_qf[a], fc.p_qf[b] = pa, 1.0 - pa

    # SF: winner distribution of (pair0 vs pair1) and (pair2 vs pair3)
    def semi(w1: dict[str, float], w2: dict[str, float]) -> dict[str, float]:
        """Winner distribution of a semifinal between two quarterfinal winners.

        Args:
            w1: Winner distribution of one quarterfinal (team -> probability).
            w2: Winner distribution of the other quarterfinal.

        Returns:
            dict[str, float]: Probability each team wins this semifinal.
        """
        out: dict[str, float] = {}
        for t, pt in w1.items():
            out[t] = out.get(t, 0.0) + pt * sum(po * p_beat(t, o) for o, po in w2.items())
        for t, pt in w2.items():
            out[t] = out.get(t, 0.0) + pt * sum(po * p_beat(t, o) for o, po in w1.items())
        return out

    sf1 = semi(qf_win[0], qf_win[1])
    sf2 = semi(qf_win[2], qf_win[3])
    # sf1 and sf2 cover disjoint halves of the bracket, so there is no key overlap.
    fc.p_final.update(sf1)
    fc.p_final.update(sf2)

    # Final: champion distribution
    for t, pt in sf1.items():
        fc.p_champion[t] = pt * sum(po * p_beat(t, o) for o, po in sf2.items())
    for t, pt in sf2.items():
        fc.p_champion[t] = pt * sum(po * p_beat(t, o) for o, po in sf1.items())

    return fc


def advance_prob_from_3way(p_win: float, p_draw: float) -> float:
    """Compute P(advance a knockout tie) from 90-minute win/draw probabilities.

    Extra time and penalties are approximated as a coin flip, so a draw after 90
    minutes contributes half its probability to advancing.

    Args:
        p_win: Probability of a win in 90 minutes.
        p_draw: Probability of a draw in 90 minutes.

    Returns:
        float: Probability of advancing past the tie.
    """
    return p_win + 0.5 * p_draw
