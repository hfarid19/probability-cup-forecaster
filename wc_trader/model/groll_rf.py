"""Hybrid random forest of Groll, Ley, Schauberger & Van Eetvelde (2019).

Faithful architecture (see GROLL_PLAN.md for the deviation ledger):
  - two rows per match, response = goals scored by the row's team;
  - covariates combine *ability parameters* from a separate time-weighted Poisson
    ranking fit (here: our Dixon-Coles attack/defense — the paper's dominant
    predictors) with FIFA rank, GDP per capita, population, host flag, confederation;
  - a random forest regresses goals -> expected goals λ per team;
  - match outcome probabilities come from an independent double-Poisson score grid.

The paper used R `ranger` (~5000 trees); we use sklearn RandomForestRegressor with
1000 trees, min_samples_leaf=5, max_features=1/3 (regression convention).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import poisson

from ..types import Outcome

CONFED_CATEGORIES = ["UEFA", "CONMEBOL", "CONCACAF", "CAF", "AFC", "OFC", "OTHER"]

NUMERIC_FEATURES = [
    "t_attack", "t_defense", "o_attack", "o_defense",
    "t_rank_points", "t_rank_pos", "o_rank_points", "o_rank_pos",
    "t_gdp_pc", "o_gdp_pc", "t_pop", "o_pop",
    "t_host", "o_host",
]


def feature_columns() -> list[str]:
    return NUMERIC_FEATURES + [f"t_confed_{c}" for c in CONFED_CATEGORIES] \
                            + [f"o_confed_{c}" for c in CONFED_CATEGORIES]


@dataclass
class TeamSnapshot:
    """Everything known about one team at the tournament freeze date."""
    attack: float
    defense: float
    rank_points: float | None
    rank_pos: float | None
    gdp_pc: float | None
    population: float | None
    host: bool
    confed: str


def team_row(t: TeamSnapshot, o: TeamSnapshot) -> dict:
    """Pure: one observation's feature dict for team `t` playing opponent `o`."""
    row: dict = {
        "t_attack": t.attack, "t_defense": t.defense,
        "o_attack": o.attack, "o_defense": o.defense,
        "t_rank_points": t.rank_points, "t_rank_pos": t.rank_pos,
        "o_rank_points": o.rank_points, "o_rank_pos": o.rank_pos,
        "t_gdp_pc": t.gdp_pc, "o_gdp_pc": o.gdp_pc,
        "t_pop": t.population, "o_pop": o.population,
        "t_host": float(t.host), "o_host": float(o.host),
    }
    for c in CONFED_CATEGORIES:
        row[f"t_confed_{c}"] = 1.0 if t.confed == c else 0.0
        row[f"o_confed_{c}"] = 1.0 if o.confed == c else 0.0
    return row


def match_rows(match_id: str, home: str, away: str, home_goals: int | None,
               away_goals: int | None, snaps: dict[str, TeamSnapshot]) -> list[dict]:
    """Pure: the two per-team observations for one match (goals None at predict time)."""
    th, ta = snaps[home], snaps[away]
    r1 = {"match_id": match_id, "team": home, "opponent": away, "goals": home_goals}
    r1.update(team_row(th, ta))
    r2 = {"match_id": match_id, "team": away, "opponent": home, "goals": away_goals}
    r2.update(team_row(ta, th))
    return [r1, r2]


def poisson_outcome_probs(lam_home: float, lam_away: float,
                          max_goals: int = 10) -> dict[Outcome, float]:
    """Independent double-Poisson score grid -> HOME/DRAW/AWAY probabilities."""
    gv = np.arange(max_goals + 1)
    grid = np.outer(poisson.pmf(gv, max(1e-3, lam_home)), poisson.pmf(gv, max(1e-3, lam_away)))
    grid /= grid.sum()
    return {
        Outcome.HOME: float(np.tril(grid, -1).sum()),
        Outcome.DRAW: float(np.trace(grid)),
        Outcome.AWAY: float(np.triu(grid, 1).sum()),
    }


@dataclass
class HybridRF:
    n_estimators: int = 1000
    min_samples_leaf: int = 5
    max_features: float = 1 / 3
    random_state: int = 42
    _model: object = field(default=None, repr=False)
    _medians: pd.Series | None = field(default=None, repr=False)

    def _matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[feature_columns()].astype(float)
        if self._medians is None:  # fit time: learn fill values
            self._medians = X.median(numeric_only=True).fillna(0.0)
        return X.fillna(self._medians)

    def fit(self, df: pd.DataFrame) -> "HybridRF":
        from sklearn.ensemble import RandomForestRegressor

        X = self._matrix(df)
        y = df["goals"].astype(float)
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators, min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features, random_state=self.random_state, n_jobs=-1,
        ).fit(X, y)
        return self

    def predict_lambda(self, df: pd.DataFrame) -> np.ndarray:
        lam = self._model.predict(self._matrix(df))
        return np.clip(lam, 0.05, None)

    def match_probabilities_from_rows(self, two_rows: pd.DataFrame) -> dict[Outcome, float]:
        """`two_rows` = the (home, away) observation pair for one match, in that order."""
        lam = self.predict_lambda(two_rows)
        return poisson_outcome_probs(lam[0], lam[1])

    def importances(self) -> pd.Series:
        return pd.Series(self._model.feature_importances_, index=feature_columns()) \
                 .sort_values(ascending=False)
