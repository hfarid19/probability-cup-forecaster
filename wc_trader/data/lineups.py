"""Per-match lineups and goal events from FIFA's live-match API (2018/2022/2026).

For every played match in a cached FIFA calendar, fetch the live-match document and
keep, per team: the match squad with starter flags (Status == 1) and positions, plus
goal events with scorer and assist ids. Cached to data/raw/lineups_<year>.json.

This is the raw material for the Probability Cup base rates (probability_cup/base_rates.py):
who actually played, and who is producing.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

from .fifa_rank import normalize_team

LIVE_URL = "https://api.fifa.com/api/v3/live/football/{comp}/{season}/{stage}/{match}?language=en"
CAL_PATH = "data/raw/fifa_calendar_{year}.json"
OUT_PATH = "data/raw/lineups_{year}.json"
STARTER_STATUS = 1


def _get(url: str) -> dict:
    """
    Fetch and decode one JSON document from the FIFA API.

    Sends a browser User-Agent because the API rejects the default urllib agent.

    Args:
        url: Fully formed API URL to request.

    Returns:
        dict: The decoded JSON response.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=45) as resp:
        return json.load(resp)


def _side(team: dict | None) -> dict | None:
    """
    Extract one team's lineup and goal events from a live-match team blob.

    Returns None when the blob is missing or has no team name (an unplayed side). Each
    player carries a starter flag (Status == STARTER_STATUS) and a position.

    Args:
        team: A HomeTeam/AwayTeam sub-document from the live-match API, or None.

    Returns:
        dict | None: {'team', 'players', 'goals'} for the side, or None if absent.
    """
    if not team or not team.get("TeamName"):
        return None
    players = []
    for p in team.get("Players") or []:
        names = p.get("PlayerName") or []
        players.append({
            "id": p.get("IdPlayer"),
            "name": names[0].get("Description") if names else None,
            "starter": p.get("Status") == STARTER_STATUS,
            "position": p.get("Position"),
        })
    goals = [{"player": g.get("IdPlayer"), "assist": g.get("IdAssistPlayer"),
              "minute": g.get("Minute")} for g in (team.get("Goals") or [])]
    return {"team": normalize_team(team["TeamName"][0]["Description"]),
            "players": players, "goals": goals}


def fetch_lineups(year: int, *, delay_s: float = 0.25, force: bool = False) -> Path:
    """
    Fetch and cache lineups and goal events for all played matches of a tournament.

    Reads the cached FIFA calendar, requests the live-match document for each played
    match, and writes the combined records to data/raw/lineups_<year>.json. Skips
    unplayed placeholders and matches whose fetch fails. Reads the cache when present
    unless force is set.

    Args:
        year: Tournament year (2018, 2022, or 2026).
        delay_s: Seconds to sleep between match requests (politeness throttle).
        force: Re-fetch even if the cache already exists.

    Returns:
        Path: The local path to the (now present) lineups JSON.
    """
    out = Path(OUT_PATH.format(year=year))
    if out.exists() and not force:
        return out
    cal = json.loads(Path(CAL_PATH.format(year=year)).read_text())
    records = []
    for m in cal["Results"]:
        if not ((m.get("Home") or {}).get("TeamName")):    # unplayed placeholder
            continue
        try:
            doc = _get(LIVE_URL.format(comp=m["IdCompetition"], season=m["IdSeason"],
                                       stage=m["IdStage"], match=m["IdMatch"]))
        except Exception as e:
            print(f"  warn: match {m.get('MatchNumber')} fetch failed: {e}")
            continue
        home, away = _side(doc.get("HomeTeam")), _side(doc.get("AwayTeam"))
        if home and away:
            records.append({"match_number": m.get("MatchNumber"),
                            "date": (m.get("Date") or "")[:10],
                            "home": home, "away": away})
        time.sleep(delay_s)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, ensure_ascii=False))
    return out


def load_lineups(year: int) -> list[dict]:
    """
    Load the cached lineups and goal events for a tournament.

    Args:
        year: Tournament year (2018, 2022, or 2026).

    Returns:
        list[dict]: One record per played match (match_number, date, home, away).
    """
    return json.loads(Path(OUT_PATH.format(year=year)).read_text())
