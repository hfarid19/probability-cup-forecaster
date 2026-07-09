"""World Bank covariates (Groll): GDP per capita and population.

Fetches `NY.GDP.PCAP.CD` and `SP.POP.TOTL` for a set of ISO3 codes via the World Bank
v2 API and caches to CSV. `value_asof` returns the latest non-null observation ≤ the
requested year (economies report with lag), within a bounded lookback.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

INDICATORS = {
    "gdp_pc": "NY.GDP.PCAP.CD",
    "population": "SP.POP.TOTL",
}
CACHE_DIR = Path("data/raw/worldbank")
_API = "https://api.worldbank.org/v2/country/{codes}/indicator/{indicator}"


def parse_worldbank_json(payload: list) -> list[dict]:
    """Pure: WB API JSON ([meta, rows]) -> [{'iso3', 'year', 'value'}, ...]."""
    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        return []
    out = []
    for row in payload[1]:
        out.append({
            "iso3": row.get("countryiso3code") or "",
            "year": int(row["date"]),
            "value": row["value"],  # may be None
        })
    return out


def fetch_indicator(iso3_codes: list[str], key: str, *, start: int = 1990, end: int = 2026,
                    cache_dir: Path = CACHE_DIR, force: bool = False) -> pd.DataFrame:
    """Fetch one indicator for many countries (single call, cached to CSV)."""
    indicator = INDICATORS[key]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{key}.csv"
    if cache.exists() and not force:
        return pd.read_csv(cache)

    url = (_API.format(codes=";".join(sorted(set(iso3_codes))), indicator=indicator)
           + f"?format=json&date={start}:{end}&per_page=20000")
    with urllib.request.urlopen(url, timeout=60) as resp:
        payload = json.load(resp)
    df = pd.DataFrame(parse_worldbank_json(payload))
    df.to_csv(cache, index=False)
    return df


class IndicatorLookup:
    """As-of lookup: latest non-null value ≤ year, within `max_lookback` years."""

    def __init__(self, df: pd.DataFrame, max_lookback: int = 8):
        self.max_lookback = max_lookback
        self._by_country: dict[str, list[tuple[int, float]]] = {}
        clean = df.dropna(subset=["value"])
        for iso, grp in clean.groupby("iso3"):
            self._by_country[str(iso)] = sorted(
                (int(r.year), float(r.value)) for r in grp.itertuples(index=False))

    def value_asof(self, iso3: str | None, year: int) -> float | None:
        if not iso3 or iso3 not in self._by_country:
            return None
        best = None
        for y, v in self._by_country[iso3]:
            if y > year:
                break
            if year - y <= self.max_lookback:
                best = v
        return best
