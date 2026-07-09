"""Scrape 2026 World Cup match odds (1X2) from OddsPortal via headless Chrome + CDP.

OddsPortal renders its results client-side from an encrypted feed, so plain HTTP won't
work; this drives your installed Chrome over the DevTools protocol: navigate to the
results pages, scroll to force lazy rendering, click through pagination, and extract
rows from the live DOM. Output joins onto the martj42 results (96/96 played matches at
time of writing) and lands in data/raw/wc2026_odds.csv, which paper_eval.py picks up
automatically as the "Market" benchmark.

    python scripts/scrape_wc2026_odds.py

Requires Google Chrome and `pip install websocket-client`. Re-run after each round.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd
import websocket

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root for wc_trader

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
RESULTS_URL = "https://www.oddsportal.com/football/world/world-championship-2026/results/"
OUT = Path("data/raw/wc2026_odds.csv")
DEBUG_PORT = 9222

# OddsPortal name -> martj42 results-dataset name
NAME_MAP = {
    "USA": "United States", "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Curacao": "Curaçao", "D.R. Congo": "DR Congo", "Korea Republic": "South Korea",
    "IR Iran": "Iran", "Czechia": "Czech Republic", "Türkiye": "Turkey",
    "Cabo Verde": "Cape Verde", "Cote d'Ivoire": "Ivory Coast",
}

# Rows are grouped under date headers; odds cells interleave desktop/mobile duplicates
# as [1,1,X,X,2,2] -> take indices 0/2/4.
EXTRACT_JS = r"""
(() => {
  const out = [];
  let curDate = null;
  const els = document.querySelectorAll('[data-testid="date-header"], [data-testid="game-row"]');
  const seen = new Set();
  for (const el of els) {
    if (el.getAttribute('data-testid') === 'date-header') { curDate = el.textContent.trim(); continue; }
    const names = [...el.querySelectorAll('.participant-name')].map(p => p.textContent.trim());
    let odds = [...el.querySelectorAll('[data-testid^="odd-container"]')].map(o => o.textContent.trim());
    const uniq = [...new Set(names)];
    if (uniq.length !== 2) continue;
    if (odds.length === 6) odds = [odds[0], odds[2], odds[4]];
    else if (odds.length > 3) odds = odds.slice(0, 3);
    const key = curDate + '|' + uniq.join('|');
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({date: curDate, home: uniq[0], away: uniq[1], odds});
  }
  return JSON.stringify(out);
})()
"""


class CDP:
    def __init__(self, ws_url: str):
        self.ws = websocket.create_connection(ws_url, timeout=120, suppress_origin=True)
        self._id = 0

    def send(self, method: str, **params):
        self._id += 1
        self.ws.send(json.dumps({"id": self._id, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self._id:
                return msg.get("result", {})


def scrape_pages(cdp: CDP, max_pages: int = 5) -> list[dict]:
    cdp.send("Page.enable")
    cdp.send("Page.navigate", url=RESULTS_URL)
    time.sleep(10)
    rows = []

    def scroll_extract():
        for _ in range(15):
            cdp.send("Runtime.evaluate", expression="window.scrollBy(0, 2000)")
            time.sleep(0.4)
        time.sleep(1.5)
        res = cdp.send("Runtime.evaluate", expression=EXTRACT_JS, returnByValue=True)
        return json.loads(res["result"]["value"])

    rows.extend(scroll_extract())
    for p in range(2, max_pages + 1):
        r = cdp.send("Runtime.evaluate", returnByValue=True, expression=(
            f"(() => {{ const el = document.querySelector('a.pagination-link[data-number=\"{p}\"]');"
            f" if (!el) return 'missing'; el.scrollIntoView(); el.click(); return 'clicked'; }})()"))
        if r["result"]["value"] != "clicked":
            break
        time.sleep(8)
        rows.extend(scroll_extract())
    return rows


def join_to_results(rows: list[dict]) -> pd.DataFrame:
    from wc_trader.data.results import load_results

    finals = [r for r in rows if "Qualification" not in r["date"]]
    fix = lambda n: NAME_MAP.get(n, n)
    by_pair: dict[frozenset, list] = {}
    for r in finals:
        try:
            o = [float(x) for x in r["odds"]]
        except ValueError:
            continue
        if len(o) == 3:
            by_pair.setdefault(frozenset((fix(r["home"]), fix(r["away"]))), []).append((fix(r["home"]), o))

    df = load_results()
    wc = df[(df.tournament == "FIFA World Cup") & (df.date.dt.year == 2026)].sort_values("date")
    matched, missing = [], []
    for m in wc.itertuples(index=False):
        cands = by_pair.get(frozenset((m.home_team, m.away_team)), [])
        if not cands:
            missing.append((m.home_team, m.away_team))
            continue
        h, o = cands[0]
        oh, ox, oa = o if h == m.home_team else (o[2], o[1], o[0])
        matched.append({"date": str(m.date.date()), "home_team": m.home_team,
                        "away_team": m.away_team, "odds_home": oh, "odds_draw": ox, "odds_away": oa})
    print(f"matched {len(matched)}/{len(wc)} played matches; missing: {missing}")
    return pd.DataFrame(matched)


def main() -> None:
    profile = Path("/tmp/wc-odds-chrome-profile")
    proc = subprocess.Popen(
        [CHROME, "--headless=new", "--disable-gpu", "--no-first-run",
         f"--remote-debugging-port={DEBUG_PORT}", f"--user-data-dir={profile}", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(3)
        targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{DEBUG_PORT}/json"))
        page = next(t for t in targets if t["type"] == "page")
        cdp = CDP(page["webSocketDebuggerUrl"])
        rows = scrape_pages(cdp)
        print(f"scraped {len(rows)} raw rows")
        out = join_to_results(rows)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(OUT, index=False)
        print(f"saved {len(out)} matches -> {OUT}")
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
