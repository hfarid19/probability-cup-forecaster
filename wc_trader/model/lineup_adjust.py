"""News-aware adjustment of the ability model from published lineups (+ player form).

Motivation (PAPER.md §5.2): the market's 2026 edge concentrates on matchday 3 and the
knockouts — segments where team news (rotations, form) exists. This layer feeds exactly
that news into the Dixon-Coles λs, using only *pre-kickoff* information: the official
starting XI (published ~1h before kickoff) and events from *previous* matches.

Two features per team-match, both zero on matchday 1 by construction:

  rotation_delta ∈ [0, 1]  — 1 − mean over today's starters of (share of the team's
      previous matches they started). A settled XI → 0; a heavily rotated XI → → 1.

  hot_delta — the team's goal over-performance so far, carried by today's starters:
      (goals scored − DC-expected goals, per match so far) × (share of the team's goal
      contributions produced by players in today's XI). A team over-performing through
      players who are on the pitch again gets a boost; rest the hot scorers and the
      boost fades. This is the "Messi is having a great tournament" term.

Adjusted expected goals for team A vs B:

  log λ'_A = log λ_A − β_own·rot_A + β_opp·rot_B + γ·hot_A

(β_own: your rotation weakens your attack; β_opp: the opponent's rotation weakens
their defense; γ: form.) The three coefficients are fit by Poisson maximum likelihood
on the 2018 + 2022 tournaments — real lineups, real goals, no post-hoc information.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

ASSIST_WEIGHT = 0.5


@dataclass
class TeamMatchNews:
    rotation_delta: float
    hot_delta: float


def _contrib(goals: list[dict]) -> dict[str, float]:
    """player id -> goals + 0.5 * assists from one match's goal events."""
    out: dict[str, float] = {}
    for g in goals:
        if g.get("player"):
            out[g["player"]] = out.get(g["player"], 0.0) + 1.0
        if g.get("assist"):
            out[g["assist"]] = out.get(g["assist"], 0.0) + ASSIST_WEIGHT
    return out


class NewsFeatureBuilder:
    """Walks one tournament's lineup records chronologically and produces, for each
    team-match, the rotation/form features using only earlier matches."""

    def __init__(self, dc, lineup_records: list[dict]):
        self.dc = dc
        self.records = sorted(lineup_records, key=lambda r: (r["date"], r["match_number"] or 0))
        # per-team running state
        self._matches: dict[str, int] = {}
        self._starts: dict[str, dict[str, int]] = {}          # team -> player -> starts
        self._contrib: dict[str, dict[str, float]] = {}       # team -> player -> g+0.5a
        self._goals_for: dict[str, float] = {}
        self._exp_goals: dict[str, float] = {}
        # one feature entry PER RECORD (repeat pairings must not overwrite each other —
        # that would leak a later match's lineup context into an earlier evaluation)
        self.per_record: list[tuple[dict, dict[str, TeamMatchNews]]] = []
        self.features: dict[tuple[str, str], list[tuple[str, dict[str, TeamMatchNews]]]] = {}
        self._build()

    def _team_news(self, team: str, starters: list[str]) -> TeamMatchNews:
        n = self._matches.get(team, 0)
        if n == 0 or not starters:
            return TeamMatchNews(0.0, 0.0)
        starts = self._starts.get(team, {})
        rotation = 1.0 - sum(starts.get(p, 0) for p in starters) / (len(starters) * n)

        contrib = self._contrib.get(team, {})
        total = sum(contrib.values())
        presence = (sum(contrib.get(p, 0.0) for p in starters) / total) if total > 0 else 1.0
        over = (self._goals_for.get(team, 0.0) - self._exp_goals.get(team, 0.0)) / n
        return TeamMatchNews(rotation_delta=rotation, hot_delta=over * presence)

    def _build(self) -> None:
        for rec in self.records:
            sh = [p["id"] for p in rec["home"]["players"] if p["starter"]]
            sa = [p["id"] for p in rec["away"]["players"] if p["starter"]]
            feats = {
                rec["home"]["team"]: self._team_news(rec["home"]["team"], sh),
                rec["away"]["team"]: self._team_news(rec["away"]["team"], sa),
            }
            self.per_record.append((rec, feats))
            key = (rec["home"]["team"], rec["away"]["team"])
            self.features.setdefault(key, []).append((rec["date"], feats))
            # update running state AFTER computing features (no lookahead)
            grid_exp = None
            try:
                g = self.dc.score_grid(rec["home"]["team"], rec["away"]["team"], neutral=True)
                gv = np.arange(g.shape[0])
                grid_exp = (float((g.sum(axis=1) * gv).sum()), float((g.sum(axis=0) * gv).sum()))
            except Exception:
                grid_exp = (1.3, 1.3)
            for side, starters, exp in ((rec["home"], sh, grid_exp[0]), (rec["away"], sa, grid_exp[1])):
                t = side["team"]
                self._matches[t] = self._matches.get(t, 0) + 1
                st = self._starts.setdefault(t, {})
                for p in starters:
                    st[p] = st.get(p, 0) + 1
                cb = self._contrib.setdefault(t, {})
                for p, c in _contrib(side["goals"]).items():
                    cb[p] = cb.get(p, 0.0) + c
                self._goals_for[t] = self._goals_for.get(t, 0.0) + sum(
                    1.0 for _ in side["goals"])
                self._exp_goals[t] = self._exp_goals.get(t, 0.0) + exp

    def news_for(self, home: str, away: str,
                 date: str | None = None) -> tuple[TeamMatchNews, TeamMatchNews] | None:
        """Features for one fixture. If the pairing occurred more than once in the
        tournament, `date` (ISO, ±2 days tolerance for timezone skew) selects the
        right occurrence; without a date, a repeated pairing raises rather than
        silently returning the wrong (possibly future) match's features."""
        entries = self.features.get((home, away))
        if not entries:
            return None
        if len(entries) == 1 and date is None:
            f = entries[0][1]
            return f[home], f[away]
        if date is None:
            raise ValueError(f"pairing {home} v {away} occurs {len(entries)}x; pass date")
        import datetime as _dt
        want = _dt.date.fromisoformat(date)
        best = min(entries, key=lambda e: abs((_dt.date.fromisoformat(e[0]) - want).days))
        if abs((_dt.date.fromisoformat(best[0]) - want).days) > 2:
            return None
        f = best[1]
        return f[home], f[away]


@dataclass
class NewsCoefficients:
    beta_own: float
    beta_opp: float
    gamma: float


def adjusted_lambdas(lam_h: float, lam_a: float, news_h: TeamMatchNews,
                     news_a: TeamMatchNews, c: NewsCoefficients) -> tuple[float, float]:
    lh = lam_h * float(np.exp(-c.beta_own * news_h.rotation_delta
                              + c.beta_opp * news_a.rotation_delta
                              + c.gamma * news_h.hot_delta))
    la = lam_a * float(np.exp(-c.beta_own * news_a.rotation_delta
                              + c.beta_opp * news_h.rotation_delta
                              + c.gamma * news_a.hot_delta))
    return lh, la


def fit_coefficients(samples: list[dict]) -> NewsCoefficients:
    """Poisson ML fit of (beta_own, beta_opp, gamma) on calibration samples:
    each sample = {lam_h, lam_a, news_h, news_a, goals_h, goals_a}."""
    def nll(x):
        c = NewsCoefficients(*x)
        total = 0.0
        for s in samples:
            lh, la = adjusted_lambdas(s["lam_h"], s["lam_a"], s["news_h"], s["news_a"], c)
            lh, la = max(1e-6, lh), max(1e-6, la)
            total -= s["goals_h"] * np.log(lh) - lh + s["goals_a"] * np.log(la) - la
        return total

    res = minimize(nll, x0=[0.3, 0.3, 0.1], method="L-BFGS-B",
                   bounds=[(-1.0, 3.0), (-1.0, 3.0), (-1.0, 1.0)])
    return NewsCoefficients(*[float(v) for v in res.x])


class LambdaGridAdapter:
    """Exposes score_grid(home, away) for any model that yields per-pair expected
    goals — e.g. the hybrid RF via a precomputed {(home, away): (λh, λa)} table.
    Lets NewsFeatureBuilder / NewsAdjustedDC wrap the RF exactly like Dixon-Coles."""

    def __init__(self, lambdas: dict[tuple[str, str], tuple[float, float]],
                 max_goals: int = 10):
        self.lambdas = lambdas
        self.max_goals = max_goals

    def score_grid(self, home: str, away: str, neutral: bool = True) -> np.ndarray:
        from scipy.stats import poisson

        lh, la = self.lambdas[(home, away)]
        gv = np.arange(self.max_goals + 1)
        g = np.outer(poisson.pmf(gv, max(1e-3, lh)), poisson.pmf(gv, max(1e-3, la)))
        return g / g.sum()


class NewsAdjustedDC:
    """Dixon-Coles whose λs are shifted by pre-kickoff lineup/form news."""

    def __init__(self, dc, builder: NewsFeatureBuilder, coef: NewsCoefficients,
                 use_form: bool = True):
        self.dc = dc
        self.builder = builder
        self.coef = coef if use_form else NewsCoefficients(coef.beta_own, coef.beta_opp, 0.0)

    def match_probabilities(self, home: str, away: str, neutral: bool = True,
                            date: str | None = None):
        from ..types import Outcome

        grid = self.dc.score_grid(home, away, neutral)
        news = self.builder.news_for(home, away, date)
        if news is not None:
            gv = np.arange(grid.shape[0])
            lam_h = float((grid.sum(axis=1) * gv).sum())
            lam_a = float((grid.sum(axis=0) * gv).sum())
            lh, la = adjusted_lambdas(lam_h, lam_a, news[0], news[1], self.coef)
            # rescale the DC grid's marginals to the adjusted means (keeps the
            # low-score correction structure, shifts the intensity)
            from scipy.stats import poisson
            ph = poisson.pmf(gv, max(1e-3, lh))
            pa = poisson.pmf(gv, max(1e-3, la))
            base_h, base_a = grid.sum(axis=1), grid.sum(axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                wh = np.where(base_h > 0, ph / base_h, 0.0)
                wa = np.where(base_a > 0, pa / base_a, 0.0)
            grid = grid * np.outer(wh, wa)
            grid = np.clip(grid, 0.0, None)
            grid /= grid.sum()
        return {
            Outcome.HOME: float(np.tril(grid, -1).sum()),
            Outcome.DRAW: float(np.trace(grid)),
            Outcome.AWAY: float(np.triu(grid, 1).sum()),
        }
