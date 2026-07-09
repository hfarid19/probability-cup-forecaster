"""Historical odds loader (Milestone 3.5).

Source: football-data.co.uk — club-league results WITH bookmaker closing odds. Free
international odds don't really exist, so we validate the *methodology* (does the model
beat the closing line and profit after costs?) on club data. The same harness accepts
international odds (e.g. Betfair historical) later — only the loader changes.

We prefer the sharpest available CLOSING line:  Pinnacle (PSC*) > market avg (AvgC*) >
Bet365 (B365C*). Closing odds are the toughest, most efficient benchmark to beat.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://www.football-data.co.uk/mmz4281"
RAW_DIR = Path("data/raw/odds")

# (home_odds, draw_odds, away_odds) column triples in order of preference.
ODDS_TRIPLES = [
    ("PSCH", "PSCD", "PSCA"),     # Pinnacle closing (sharpest)
    ("AvgCH", "AvgCD", "AvgCA"),  # market average closing
    ("B365CH", "B365CD", "B365CA"),
    ("PSH", "PSD", "PSA"),        # fall back to opening if no closing
    ("B365H", "B365D", "B365A"),
]


def fetch_league_season(season: str, league: str, force: bool = False) -> Path | None:
    """Download one league-season CSV (e.g. season='2324', league='E0'). Returns path."""
    out = RAW_DIR / f"{season}_{league}.csv"
    if out.exists() and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(f"{BASE}/{season}/{league}.csv", out)
        return out
    except Exception:
        return None


def _pick_odds(df: pd.DataFrame) -> tuple[str, str, str] | None:
    for triple in ODDS_TRIPLES:
        if all(c in df.columns for c in triple):
            return triple
    return None


def load_odds(seasons: list[str], leagues: list[str]) -> pd.DataFrame:
    """Fetch + normalize multiple league-seasons into one DataFrame with de-vigged
    market probabilities (market_home/market_draw/market_away)."""
    frames = []
    for season in seasons:
        for league in leagues:
            path = fetch_league_season(season, league)
            if path is None:
                continue
            raw = pd.read_csv(path, encoding="latin-1")
            triple = _pick_odds(raw)
            if triple is None or not {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}.issubset(raw.columns):
                continue
            h, d, a = triple
            sub = raw[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", h, d, a]].copy()
            sub.columns = ["date", "home_team", "away_team", "home_score", "away_score",
                           "odds_home", "odds_draw", "odds_away"]
            sub["league"] = league
            frames.append(sub)

    if not frames:
        raise RuntimeError("No odds data could be loaded — check connectivity / seasons.")

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["odds_home", "odds_draw", "odds_away", "home_score", "away_score"])
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = False  # club league games are at real home venues

    # De-vig: invert odds to raw probs, then normalize so H+D+A = 1 (removes the overround).
    inv = 1.0 / df[["odds_home", "odds_draw", "odds_away"]].to_numpy()
    inv = inv / inv.sum(axis=1, keepdims=True)
    df["market_home"], df["market_draw"], df["market_away"] = inv[:, 0], inv[:, 1], inv[:, 2]
    df["overround"] = (1.0 / df[["odds_home", "odds_draw", "odds_away"]].to_numpy()).sum(axis=1)

    return df.sort_values("date").reset_index(drop=True)


DEFAULT_SEASONS = ["1819", "1920", "2021", "2122", "2223", "2324", "2425"]
DEFAULT_LEAGUES = ["E0", "E1", "D1", "I1", "SP1", "F1"]  # top divisions, 5 countries
