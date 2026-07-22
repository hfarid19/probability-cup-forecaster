"""World Bank covariates (Groll): GDP per capita and population.

Fetches `NY.GDP.PCAP.CD` and `SP.POP.TOTL` for a set of ISO3 codes via the World Bank
v2 API and caches to CSV. `value_asof` returns the latest non-null observation on or
before the requested year (economies report with lag), within a bounded lookback.
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
    """
    Convert a World Bank API JSON response into flat observation rows.

    The API returns [metadata, rows]; a missing or empty rows element yields an empty
    list. Values may be None (missing observation) and are preserved as-is.

    Args:
        payload: Decoded JSON from the World Bank v2 indicator endpoint.

    Returns:
        list[dict]: Rows of {'iso3', 'year', 'value'} (value may be None).
    """
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
    """
    Fetch one indicator for many countries in a single API call, cached to CSV.

    Reads the cache when present unless force is set; otherwise queries the World Bank
    API for the given ISO3 codes and year range and writes the CSV.

    Args:
        iso3_codes: ISO3 country codes to request.
        key: Indicator key into INDICATORS ("gdp_pc" or "population").
        start: First year of the requested range.
        end: Last year of the requested range.
        cache_dir: Directory holding the per-indicator cache CSVs.
        force: Re-fetch even if the cache already exists.

    Returns:
        pd.DataFrame: Observations with columns iso3, year, value.
    """
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
    """
    As-of lookup for a World Bank indicator (no-hindsight freeze rule).

    Returns the latest non-null value on or before a year, but only within max_lookback
    years, so a stale figure is never carried forward indefinitely and no future
    observation leaks into a feature.
    """

    def __init__(self, df: pd.DataFrame, max_lookback: int = 8):
        """
        Index observations by country for as-of lookups.

        Args:
            df: Observations with columns iso3, year, value (null values are dropped).
            max_lookback: Maximum age in years of a value that may be carried forward.
        """
        self.max_lookback = max_lookback
        self._by_country: dict[str, list[tuple[int, float]]] = {}
        clean = df.dropna(subset=["value"])
        for iso, grp in clean.groupby("iso3"):
            self._by_country[str(iso)] = sorted(
                (int(r.year), float(r.value)) for r in grp.itertuples(index=False))

    def value_asof(self, iso3: str | None, year: int) -> float | None:
        """
        Return the latest non-null value on or before year, within max_lookback.

        Args:
            iso3: ISO3 country code, or None for an unmapped team.
            year: The as-of year to look up.

        Returns:
            float | None: The most recent eligible value, or None if none qualifies.
        """
        if not iso3 or iso3 not in self._by_country:
            return None
        best = None
        for y, v in self._by_country[iso3]:
            if y > year:
                break
            if year - y <= self.max_lookback:
                best = v
        return best
