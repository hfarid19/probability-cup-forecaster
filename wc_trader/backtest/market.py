"""Model-vs-market backtest (Milestone 3.5) — the decisive test.

Two questions, both answered here:
  1. Calibration vs the line: is the model's log-loss <= the closing line's log-loss?
     (Beating a sharp closing line is HARD; matching it is already a good result.)
  2. Does it make money? A value-betting simulation: back any outcome whose expected
     value at the OFFERED odds clears a threshold, settle on the result, apply exchange
     commission, and report ROI.

Note: bookmaker closing odds already include the vig, and we ALSO charge exchange
commission — so this PnL is deliberately conservative vs. what real Betfair prices give.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..model.dixon_coles import DixonColesModel
from ..model.elo import EloModel
from ..types import Outcome
from .metrics import log_loss
from .simulator import CLASSES, match_outcome

_COL = {"HOME": "market_home", "DRAW": "market_draw", "AWAY": "market_away"}
_ODDS = {"HOME": "odds_home", "DRAW": "odds_draw", "AWAY": "odds_away"}


@dataclass
class MarketReport:
    n_eval: int
    model_log_loss: float
    market_log_loss: float
    # value betting
    n_bets: int
    total_staked: float
    profit: float
    roi: float
    hit_rate: float
    edge_threshold: float
    commission: float

    @property
    def beats_line(self) -> bool:
        return self.model_log_loss < self.market_log_loss


def run_elo_vs_market(
    df: pd.DataFrame,
    *,
    eval_start: str | None = None,
    home_advantage: float = 65.0,
    draw_factor: float = 0.27,
    k: float = 20.0,
    edge_threshold: float = 0.05,   # min EV at offered odds to place a value bet
    commission: float = 0.02,
) -> MarketReport:
    """Walk-forward Elo (online) scored against the closing line + a value-bet sim."""
    model = EloModel(home_advantage=home_advantage, draw_factor=draw_factor, k=k)
    eval_ts = pd.Timestamp(eval_start) if eval_start else None

    model_probs: list[dict] = []
    market_probs: list[dict] = []
    outcomes: list[str] = []

    n_bets = staked = profit = wins = 0.0

    for r in df.itertuples(index=False):
        in_eval = eval_ts is None or r.date >= eval_ts
        if in_eval:
            p = model.match_probabilities(r.home_team, r.away_team, bool(r.neutral))
            mp = {c: p[Outcome(c)] for c in CLASSES}
            market = {c: getattr(r, _COL[c]) for c in CLASSES}
            result = match_outcome(r.home_score, r.away_score)

            model_probs.append(mp)
            market_probs.append(market)
            outcomes.append(result)

            # Value bets: back any outcome with positive EV beyond the threshold.
            for c in CLASSES:
                odds = getattr(r, _ODDS[c])
                ev = mp[c] * odds - 1.0          # expected profit per unit staked
                if ev > edge_threshold:
                    n_bets += 1
                    staked += 1.0
                    if c == result:
                        profit += (odds - 1.0) * (1.0 - commission)
                        wins += 1
                    else:
                        profit -= 1.0
        model.update(r.home_team, r.away_team, r.home_score, r.away_score, bool(r.neutral))

    if not outcomes:
        raise ValueError("No matches in the evaluation window.")

    return MarketReport(
        n_eval=len(outcomes),
        model_log_loss=log_loss(model_probs, outcomes),
        market_log_loss=log_loss(market_probs, outcomes),
        n_bets=int(n_bets),
        total_staked=staked,
        profit=profit,
        roi=(profit / staked) if staked else 0.0,
        hit_rate=(wins / n_bets) if n_bets else 0.0,
        edge_threshold=edge_threshold,
        commission=commission,
    )


def run_dc_vs_market(
    df: pd.DataFrame,
    *,
    eval_start: str | None = None,
    refit_days: int = 120,
    train_window_years: int = 3,
    half_life_days: float = 360.0,
    min_matches: int = 8,
    edge_threshold: float = 0.05,
    commission: float = 0.02,
) -> MarketReport:
    """Dixon-Coles (periodic refit) scored against the closing line + value-bet sim."""
    eval_ts = pd.Timestamp(eval_start) if eval_start else df.date.min()
    dc: DixonColesModel | None = None
    last_fit: pd.Timestamp | None = None

    model_probs: list[dict] = []
    market_probs: list[dict] = []
    outcomes: list[str] = []
    n_bets = staked = profit = wins = 0.0

    for r in df.itertuples(index=False):
        if r.date < eval_ts:
            continue
        if dc is None or (r.date - last_fit).days >= refit_days:
            lo = r.date - pd.DateOffset(years=train_window_years)
            train = df[(df.date >= lo) & (df.date < r.date)]
            dc = DixonColesModel().fit(train, half_life_days=half_life_days, min_matches=min_matches)
            last_fit = r.date
        if r.home_team not in dc.attack or r.away_team not in dc.attack:
            continue

        p = dc.match_probabilities(r.home_team, r.away_team, bool(r.neutral))
        mp = {c: p[Outcome(c)] for c in CLASSES}
        market = {c: getattr(r, _COL[c]) for c in CLASSES}
        result = match_outcome(r.home_score, r.away_score)
        model_probs.append(mp)
        market_probs.append(market)
        outcomes.append(result)

        for c in CLASSES:
            odds = getattr(r, _ODDS[c])
            if mp[c] * odds - 1.0 > edge_threshold:
                n_bets += 1
                staked += 1.0
                if c == result:
                    profit += (odds - 1.0) * (1.0 - commission)
                    wins += 1
                else:
                    profit -= 1.0

    if not outcomes:
        raise ValueError("No matches in the evaluation window.")

    return MarketReport(
        n_eval=len(outcomes),
        model_log_loss=log_loss(model_probs, outcomes),
        market_log_loss=log_loss(market_probs, outcomes),
        n_bets=int(n_bets),
        total_staked=staked,
        profit=profit,
        roi=(profit / staked) if staked else 0.0,
        hit_rate=(wins / n_bets) if n_bets else 0.0,
        edge_threshold=edge_threshold,
        commission=commission,
    )
