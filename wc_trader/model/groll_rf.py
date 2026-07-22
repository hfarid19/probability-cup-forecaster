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
    """List the model's feature column names in a fixed, stable order.

    Combines the numeric features with the one-hot confederation columns for both
    the team ("t_") and the opponent ("o_"). The order defines the training and
    prediction matrix layout, so it must stay consistent.

    Returns:
        list[str]: Ordered feature column names.
    """
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
    """Build the feature dict for one team facing one opponent.

    Pure function: lays out the team's own attributes under "t_" keys and the
    opponent's under "o_" keys, including one-hot confederation flags.

    Args:
        t: Snapshot of the team whose goals are being modeled.
        o: Snapshot of the opposing team.

    Returns:
        dict: Feature name -> value for this single observation.
    """
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
    """Build both per-team observation rows for a single match.

    Pure function: the model uses two rows per match, one per team, each with the
    goals that team scored as the response. Goals are None at prediction time.

    Args:
        match_id: Identifier attached to both rows so they can be paired later.
        home: Home team name.
        away: Away team name.
        home_goals: Goals scored by the home team, or None when predicting.
        away_goals: Goals scored by the away team, or None when predicting.
        snaps: Team name -> snapshot lookup for both teams.

    Returns:
        list[dict]: Two rows, the home team's observation followed by the away team's.
    """
    th, ta = snaps[home], snaps[away]
    r1 = {"match_id": match_id, "team": home, "opponent": away, "goals": home_goals}
    r1.update(team_row(th, ta))
    r2 = {"match_id": match_id, "team": away, "opponent": home, "goals": away_goals}
    r2.update(team_row(ta, th))
    return [r1, r2]


def poisson_outcome_probs(lam_home: float, lam_away: float,
                          max_goals: int = 10) -> dict[Outcome, float]:
    """Convert two expected-goals rates into outcome probabilities.

    Builds an independent double-Poisson score grid from the two rates (each
    floored at a tiny positive value) and sums it into HOME/DRAW/AWAY.

    Args:
        lam_home: Expected goals for the home team.
        lam_away: Expected goals for the away team.
        max_goals: Highest goal count per team included in the grid.

    Returns:
        dict[Outcome, float]: Probabilities keyed by HOME/DRAW/AWAY summing to ~1.0.
    """
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
    """Random forest that regresses per-team goals onto match features.

    Wraps an sklearn RandomForestRegressor with the hyperparameters from Groll et
    al. (2019). It learns median fill values for missing features at fit time and
    reuses them at predict time, then turns predicted expected goals into outcome
    probabilities via an independent double-Poisson grid.
    """

    n_estimators: int = 1000
    min_samples_leaf: int = 5
    max_features: float = 1 / 3
    random_state: int = 42
    _model: object = field(default=None, repr=False)
    _medians: pd.Series | None = field(default=None, repr=False)

    def _matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and clean the feature matrix, filling missing values.

        On the first call (fit time) it learns per-column median fill values;
        later calls reuse them so training and prediction fill consistently.

        Args:
            df: Rows containing at least the feature columns.

        Returns:
            pd.DataFrame: Float feature matrix with missing values filled.
        """
        X = df[feature_columns()].astype(float)
        if self._medians is None:  # fit time: learn fill values
            self._medians = X.median(numeric_only=True).fillna(0.0)
        return X.fillna(self._medians)

    def fit(self, df: pd.DataFrame) -> "HybridRF":
        """Train the random forest on per-team goal observations.

        Args:
            df: Rows with the feature columns plus a "goals" response column.

        Returns:
            HybridRF: This instance, now fitted (for chaining).
        """
        from sklearn.ensemble import RandomForestRegressor

        X = self._matrix(df)
        y = df["goals"].astype(float)
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators, min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features, random_state=self.random_state, n_jobs=-1,
        ).fit(X, y)
        return self

    def predict_lambda(self, df: pd.DataFrame) -> np.ndarray:
        """Predict expected goals (lambda) for each row, floored above zero.

        Args:
            df: Rows containing the feature columns.

        Returns:
            np.ndarray: Predicted expected goals per row, clipped to at least 0.05.
        """
        lam = self._model.predict(self._matrix(df))
        return np.clip(lam, 0.05, None)

    def match_probabilities_from_rows(self, two_rows: pd.DataFrame) -> dict[Outcome, float]:
        """Predict outcome probabilities for a match from its two team rows.

        Args:
            two_rows: The (home, away) observation pair for one match, in that order.

        Returns:
            dict[Outcome, float]: Probabilities keyed by HOME/DRAW/AWAY summing to ~1.0.
        """
        lam = self.predict_lambda(two_rows)
        return poisson_outcome_probs(lam[0], lam[1])

    def importances(self) -> pd.Series:
        """Return the forest's feature importances, sorted high to low.

        Returns:
            pd.Series: Importance per feature, indexed by feature name, descending.
        """
        return pd.Series(self._model.feature_importances_, index=feature_columns()) \
                 .sort_values(ascending=False)


def rf_lambda_table(rf: HybridRF, snaps: dict) -> dict[tuple, tuple[float, float]]:
    """Predict expected goals for every ordered team pairing in one batch.

    Assembles the two rows for each distinct (team, opponent) pair, runs a single
    sklearn prediction, and unpacks the results back into per-pair goal rates.

    Args:
        rf: A fitted HybridRF used to predict expected goals.
        snaps: Team name -> TeamSnapshot lookup for all teams to include.

    Returns:
        dict[tuple, tuple[float, float]]: (team, opponent) -> (team lambda, opponent
        lambda) for every ordered pair of distinct teams.
    """
    teams = sorted(snaps)
    rows, pairs = [], []
    for a in teams:
        for b in teams:
            if a == b:
                continue
            rows.extend(match_rows(f"{a}|{b}", a, b, None, None, snaps))
            pairs.append((a, b))
    lam = rf.predict_lambda(pd.DataFrame(rows))
    return {p: (float(lam[2 * i]), float(lam[2 * i + 1])) for i, p in enumerate(pairs)}
