"""Historical FIFA rankings loader (Groll covariate).

Two sources, merged by `load_fifa_rankings`:
  1. Dato-Futbol `ranking_fifa_historical.csv` — official FIFA releases scraped from
     fifa.com, Dec 1992 → Sep 2024.
  2. A supplement for the official **11 June 2026 pre-tournament release**, recovered
     from FIFA's public live-ranking API: `PrevRank`/`PrevPoints` on
     api.fifa.com/api/v3/fifarankings/rankings/live are the last *official* release
     before the live (in-tournament) updates began — i.e. exactly the freeze-date
     snapshot. (The live Rank/TotalPoints fields include tournament results and must
     NOT be used — lookahead.) Fetch via `fetch_prewc2026_snapshot()`.

Releases between Oct 2025 and Apr 2026 remain absent, which is irrelevant here: we only
query as-of World Cup start dates, all of which are covered by source 1 + the supplement.

FIFA naming differs from the martj42 results dataset ("Korea Republic" vs "South
Korea"); `_NAME_MAP` harmonizes to results-dataset names.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

RANK_URL = ("https://raw.githubusercontent.com/Dato-Futbol/fifa-ranking/master/"
            "ranking_fifa_historical.csv")
DEFAULT_PATH = Path("data/raw/fifa_ranking.csv")

LIVE_API = "https://api.fifa.com/api/v3/fifarankings/rankings/live?gender=1&sportType=0&language=en"
SUPPLEMENT_PATH = Path("data/raw/fifa_ranking_2026.csv")
PREWC2026_RELEASE_DATE = "2026-06-11"

# FIFA-style name -> martj42 results-dataset name.
_NAME_MAP = {
    "USA": "United States",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Congo DR": "DR Congo",
    "Cabo Verde": "Cape Verde",
    "China PR": "China",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "FYR Macedonia": "North Macedonia",
    "Kyrgyz Republic": "Kyrgyzstan",
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "St. Lucia": "Saint Lucia",
    "St. Vincent / Grenadines": "Saint Vincent and the Grenadines",
}


def fetch_fifa_rankings(path: str | Path = DEFAULT_PATH, url: str = RANK_URL,
                        force: bool = False) -> Path:
    p = Path(path)
    if p.exists() and not force:
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, p)
    return p


def normalize_team(name: str) -> str:
    name = name.replace(" (unranked)", "").strip()
    return _NAME_MAP.get(name, name)


def parse_live_rankings(payload: dict, release_date: str = PREWC2026_RELEASE_DATE) -> list[dict]:
    """Pure: live-API JSON -> [{'team', 'date', 'points'}] for the last OFFICIAL release.

    Uses PrevPoints (the pre-tournament official release), never the live TotalPoints.
    Teams with no previous official points are skipped.
    """
    rows = []
    for r in payload.get("Results", []):
        names = r.get("TeamName") or []
        name = names[0].get("Description") if names else None
        pts = r.get("PrevPoints")
        if not name or pts is None:
            continue
        rows.append({"team": name, "date": release_date, "points": float(pts)})
    return rows


def fetch_prewc2026_snapshot(path: str | Path = SUPPLEMENT_PATH, url: str = LIVE_API,
                             force: bool = False) -> Path:
    """Fetch + cache the official 2026-06-11 release recovered from the live API."""
    p = Path(path)
    if p.exists() and not force:
        return p
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.load(resp)
    rows = parse_live_rankings(payload)
    if len(rows) < 150:  # sanity: expect ~211 ranked teams
        raise RuntimeError(f"Live ranking API returned only {len(rows)} teams — refusing to cache.")
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def load_fifa_rankings(path: str | Path = DEFAULT_PATH,
                       supplement: str | Path | None = SUPPLEMENT_PATH) -> pd.DataFrame:
    """Return tidy frame (team, date, points, rank) with harmonized names.

    Merges the historical CSV with the 2026 supplement when present. `rank` is the
    position within each release (1 = best), recomputed from points so it is consistent
    across both sources.
    """
    df = pd.read_csv(path)
    df = df.rename(columns={"total_points": "points"})[["team", "date", "points"]]
    sup = Path(supplement) if supplement else None
    if sup and sup.exists():
        df = pd.concat([df, pd.read_csv(sup)[["team", "date", "points"]]], ignore_index=True)
    df["team"] = df["team"].astype(str).map(normalize_team)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["points"])
    df["rank"] = df.groupby("date")["points"].rank(ascending=False, method="min").astype(int)
    return df.sort_values(["date", "rank"]).reset_index(drop=True)


class RankLookup:
    """As-of lookup: latest release ≤ a given date, O(1) per team after slicing."""

    def __init__(self, df: pd.DataFrame):
        self._df = df
        self._releases = df["date"].drop_duplicates().sort_values().reset_index(drop=True)

    def release_asof(self, date: str | pd.Timestamp) -> pd.Timestamp | None:
        ts = pd.Timestamp(date)
        eligible = self._releases[self._releases <= ts]
        return None if eligible.empty else eligible.iloc[-1]

    def table_asof(self, date: str | pd.Timestamp) -> dict[str, tuple[float, int]]:
        """{team: (points, rank)} at the latest release ≤ date. Empty dict if none."""
        rel = self.release_asof(date)
        if rel is None:
            return {}
        snap = self._df[self._df["date"] == rel]
        return {r.team: (float(r.points), int(r.rank)) for r in snap.itertuples(index=False)}

    def staleness_days(self, date: str | pd.Timestamp) -> int | None:
        rel = self.release_asof(date)
        return None if rel is None else int((pd.Timestamp(date) - rel).days)
