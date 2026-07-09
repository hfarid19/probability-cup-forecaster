"""Scrape 2022 World Cup match odds (1X2, average) from BetExplorer.

BetExplorer's archive pages are server-rendered — plain HTTP works (unlike OddsPortal).
The 2022 tournament lives at /football/world/world-cup-2022/results/ with two finals
stages: 'Main' (group stage) and 'Play Offs' (knockout). Rows carry `data-odd`
attributes with the average market odds.

    python scripts/scrape_wc2022_odds.py    ->  data/raw/wc2022_odds.csv
"""
from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root for wc_trader

BASE = "https://www.betexplorer.com/football/world/world-cup-2022/results/"
STAGES = {"group": "?stage=zkyDYRLU", "knockout": "?stage=823QwKIu"}
OUT = Path("data/raw/wc2022_odds.csv")
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

# BetExplorer name -> martj42 results-dataset name
NAME_MAP = {"USA": "United States", "Korea Republic": "South Korea", "IR Iran": "Iran"}


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", "replace")


def parse_rows(html: str) -> list[dict]:
    """Each result row: teams from the in-match link, then three data-odd cells."""
    out = []
    # the winner's name is wrapped in <strong> on BetExplorer result rows
    team_pat = (r'class="in-match"><span>(?:<strong>)?([^<]+)(?:</strong>)?</span> - '
                r'<span>(?:<strong>)?([^<]+)(?:</strong>)?</span>')
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
        teams = re.findall(team_pat, tr)
        odds = re.findall(r'data-odd="([\d.]+)"', tr)
        if len(teams) == 1 and len(odds) >= 3:
            h, a = (NAME_MAP.get(t.strip(), t.strip()) for t in teams[0])
            out.append({"home": h, "away": a,
                        "odds_home": float(odds[0]), "odds_draw": float(odds[1]),
                        "odds_away": float(odds[2])})
    return out


def main() -> None:
    rows = []
    for stage, q in STAGES.items():
        html = fetch(BASE + q)
        got = parse_rows(html)
        print(f"{stage}: {len(got)} matches")
        rows.extend(got)

    from wc_trader.data.results import load_results
    df = load_results()
    wc = df[(df.tournament == "FIFA World Cup") & (df.date.dt.year == 2022)].sort_values("date")

    by_pair = {frozenset((r["home"], r["away"])): r for r in rows}
    matched, missing = [], []
    for m in wc.itertuples(index=False):
        r = by_pair.get(frozenset((m.home_team, m.away_team)))
        if r is None:
            missing.append((m.home_team, m.away_team))
            continue
        if r["home"] == m.home_team:
            oh, ox, oa = r["odds_home"], r["odds_draw"], r["odds_away"]
        else:
            oh, ox, oa = r["odds_away"], r["odds_draw"], r["odds_home"]
        matched.append({"date": str(m.date.date()), "home_team": m.home_team,
                        "away_team": m.away_team, "odds_home": oh, "odds_draw": ox,
                        "odds_away": oa})

    print(f"matched {len(matched)}/{len(wc)}; missing: {missing}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matched).to_csv(OUT, index=False)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
