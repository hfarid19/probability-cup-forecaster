"""Per-match lineups and goal events from FIFA's live-match API (2018/2022/2026).

For every played match in a cached FIFA calendar, fetch the live-match document and
keep, per team: the match squad with starter flags (Status == 1) and positions, plus
goal events with scorer and assist ids. Cached to data/raw/lineups_<year>.json.

This is the raw material for the lineup-strength and player-form adjustments
(wc_trader/model/lineup_adjust.py): who actually played, and who is producing.
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
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=45) as resp:
        return json.load(resp)


def _side(team: dict | None) -> dict | None:
    """Extract one team's lineup + goals from a live-match team blob (pure-ish)."""
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
    """Fetch + cache lineups for all played matches of one tournament."""
    out = Path(OUT_PATH.format(year=year))
    if out.exists() and not force:
        return out
    cal = json.loads(Path(CAL_PATH.format(year=year)).read_text())
    records = []
    for m in cal["Results"]:
        if not ((m.get("Home") or {}).get("TeamName")):    # unplayed placeholder
            continue
        try:
            d = _get(LIVE_URL.format(comp=m["IdCompetition"], season=m["IdSeason"],
                                     stage=m["IdStage"], match=m["IdMatch"]))
        except Exception as e:
            print(f"  warn: match {m.get('MatchNumber')} fetch failed: {e}")
            continue
        home, away = _side(d.get("HomeTeam")), _side(d.get("AwayTeam"))
        if home and away:
            records.append({"match_number": m.get("MatchNumber"),
                            "date": (m.get("Date") or "")[:10],
                            "home": home, "away": away})
        time.sleep(delay_s)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, ensure_ascii=False))
    return out


def load_lineups(year: int) -> list[dict]:
    return json.loads(Path(OUT_PATH.format(year=year)).read_text())
