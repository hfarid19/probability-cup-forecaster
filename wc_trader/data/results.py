"""Historical international match results loader (Milestone 2).

Source: martj42/international_results (free, ~50k internationals 1872-present).
Schema: date, home_team, away_team, home_score, away_score, tournament, city,
country, neutral. Used to fit/evaluate Elo and Dixon-Coles.

    from wc_trader.data.results import fetch_results, load_results
    fetch_results()                 # downloads to data/raw/results.csv (once)
    df = load_results()
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["date", "home_team", "away_team", "home_score", "away_score"]
DEFAULT_PATH = Path("data/raw/results.csv")
SOURCE_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"


def fetch_results(path: str | Path = DEFAULT_PATH, url: str = SOURCE_URL, force: bool = False) -> Path:
    """Download the results dataset if not already present. Returns the local path."""
    p = Path(path)
    if p.exists() and not force:
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, p)
    return p


def load_results(path: str | Path = DEFAULT_PATH) -> pd.DataFrame:
    """Load + clean the results CSV into a chronologically sorted DataFrame."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Results dataset not found at {p}. Run fetch_results() first.")
    df = pd.read_csv(p)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"])
    # Drop rows without a recorded score (future fixtures / abandoned).
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    if "neutral" not in df.columns:
        df["neutral"] = False
    df["neutral"] = df["neutral"].astype(bool)
    return df.sort_values("date").reset_index(drop=True)
