"""Historical FIFA rankings loader (Groll covariate).

Two sources, merged by `load_fifa_rankings`:
  1. Dato-Futbol `ranking_fifa_historical.csv`, official FIFA releases scraped from
     fifa.com, Dec 1992 to Sep 2024.
  2. A supplement for the official 11 June 2026 pre-tournament release, recovered
     from FIFA's public live-ranking API: `PrevRank`/`PrevPoints` on
     api.fifa.com/api/v3/fifarankings/rankings/live are the last official release
     before the live (in-tournament) updates began, i.e. exactly the freeze-date
     snapshot. (The live Rank/TotalPoints fields include tournament results and must
     NOT be used: that would be lookahead.) Fetch via `fetch_prewc2026_snapshot()`.

Releases between Oct 2025 and Apr 2026 remain absent, which is irrelevant here: we only
query as-of World Cup start dates, all of which are covered by source 1 plus the supplement.

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
    """
    Download the Dato-Futbol historical FIFA rankings CSV if not already present.

    Hits the network only when the file is missing or force is set.

    Args:
        path: Local path to read from, or write the downloaded CSV to.
        url: Source URL of the historical rankings CSV.
        force: Re-download even if the local file already exists.

    Returns:
        Path: The local path to the (now present) rankings CSV.
    """
    p = Path(path)
    if p.exists() and not force:
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, p)
    return p


def normalize_team(name: str) -> str:
    """
    Harmonize a FIFA-style team name to the martj42 results-dataset spelling.

    Strips an "(unranked)" suffix and applies the _NAME_MAP overrides so names join
    cleanly against the results data. Unmapped names pass through unchanged.

    Args:
        name: A team name as it appears in a FIFA source.

    Returns:
        str: The results-dataset spelling of the team name.
    """
    name = name.replace(" (unranked)", "").strip()
    return _NAME_MAP.get(name, name)


def parse_live_rankings(payload: dict, release_date: str = PREWC2026_RELEASE_DATE) -> list[dict]:
    """
    Extract the last official ranking release from the FIFA live-API payload.

    Reads PrevPoints (the pre-tournament official release), never the live TotalPoints,
    to avoid using in-tournament results (lookahead). Teams with no previous official
    points are skipped.

    Args:
        payload: Decoded JSON from the FIFA live-rankings API.
        release_date: Date to stamp on each row (the official release date).

    Returns:
        list[dict]: Rows of {'team', 'date', 'points'} for the last official release.
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
    """
    Fetch and cache the official 2026-06-11 FIFA release from the live API.

    Recovers the freeze-date snapshot from PrevPoints and refuses to cache a
    suspiciously small response (expects roughly 211 ranked teams). Hits the network
    only when the cache is missing or force is set.

    Args:
        path: Local path to read from, or write the cached snapshot CSV to.
        url: FIFA live-rankings API URL.
        force: Re-fetch even if the cache already exists.

    Returns:
        Path: The local path to the (now present) snapshot CSV.
    """
    p = Path(path)
    if p.exists() and not force:
        return p
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.load(resp)
    rows = parse_live_rankings(payload)
    if len(rows) < 150:  # sanity: expect ~211 ranked teams
        raise RuntimeError(f"Live ranking API returned only {len(rows)} teams. Refusing to cache.")
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def load_fifa_rankings(path: str | Path = DEFAULT_PATH,
                       supplement: str | Path | None = SUPPLEMENT_PATH) -> pd.DataFrame:
    """
    Load a tidy rankings frame (team, date, points, rank) with harmonized names.

    Merges the historical CSV with the 2026 supplement when present. `rank` is the
    position within each release (1 = best), recomputed from points so it is consistent
    across both sources.

    Args:
        path: Local path to the historical rankings CSV.
        supplement: Optional path to the 2026 supplement CSV, or None to skip it.

    Returns:
        pd.DataFrame: Rows of (team, date, points, rank) sorted by date then rank.
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
    """
    As-of lookup for FIFA ranking releases (no-hindsight freeze rule).

    Given a date, returns the latest official release on or before it, so features never
    peek at rankings published after a match. Releases are pre-sorted once; each per-team
    lookup is cheap after slicing to the chosen release.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Index the rankings frame by release date for as-of lookups.

        Args:
            df: Tidy rankings frame from load_fifa_rankings (team, date, points, rank).
        """
        self._df = df
        self._releases = df["date"].drop_duplicates().sort_values().reset_index(drop=True)

    def release_asof(self, date: str | pd.Timestamp) -> pd.Timestamp | None:
        """
        Return the latest release date on or before the given date.

        Args:
            date: The as-of date (freeze date) to look up.

        Returns:
            pd.Timestamp | None: The newest release on or before date, or None if no
                release predates it.
        """
        ts = pd.Timestamp(date)
        eligible = self._releases[self._releases <= ts]
        return None if eligible.empty else eligible.iloc[-1]

    def table_asof(self, date: str | pd.Timestamp) -> dict[str, tuple[float, int]]:
        """
        Return the ranking table at the latest release on or before the given date.

        Args:
            date: The as-of date (freeze date) to look up.

        Returns:
            dict[str, tuple[float, int]]: {team: (points, rank)} at that release, or an
                empty dict if no release predates the date.
        """
        rel = self.release_asof(date)
        if rel is None:
            return {}
        snap = self._df[self._df["date"] == rel]
        return {r.team: (float(r.points), int(r.rank)) for r in snap.itertuples(index=False)}

    def staleness_days(self, date: str | pd.Timestamp) -> int | None:
        """
        Return how many days old the as-of release is relative to the given date.

        Args:
            date: The as-of date (freeze date) to measure against.

        Returns:
            int | None: Days between the date and the release used, or None if no
                release predates it.
        """
        rel = self.release_asof(date)
        return None if rel is None else int((pd.Timestamp(date) - rel).days)
