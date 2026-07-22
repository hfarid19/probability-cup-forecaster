"""Static World Cup metadata: editions/hosts, confederations, and ISO3 codes.

Team names follow the martj42 results dataset conventions ("United States",
"South Korea", "Ivory Coast", ...). Anything not mapped falls back explicitly
("OTHER" confederation / None ISO3) and is logged by callers (fail soft, visibly).
"""
from __future__ import annotations

# tournament year -> (start date, host teams). Freeze date for features = start.
WORLD_CUPS: dict[int, dict] = {
    1998: {"start": "1998-06-10", "hosts": ["France"]},
    2002: {"start": "2002-05-31", "hosts": ["South Korea", "Japan"]},
    2006: {"start": "2006-06-09", "hosts": ["Germany"]},
    2010: {"start": "2010-06-11", "hosts": ["South Africa"]},
    2014: {"start": "2014-06-12", "hosts": ["Brazil"]},
    2018: {"start": "2018-06-14", "hosts": ["Russia"]},
    2022: {"start": "2022-11-20", "hosts": ["Qatar"]},
    2026: {"start": "2026-06-11", "hosts": ["United States", "Canada", "Mexico"]},
}

CONFEDERATIONS: dict[str, str] = {
    # UEFA
    "England": "UEFA", "Scotland": "UEFA", "Wales": "UEFA", "Northern Ireland": "UEFA",
    "Republic of Ireland": "UEFA", "Ireland": "UEFA", "France": "UEFA", "Germany": "UEFA",
    "Italy": "UEFA", "Spain": "UEFA", "Portugal": "UEFA", "Netherlands": "UEFA",
    "Belgium": "UEFA", "Croatia": "UEFA", "Serbia": "UEFA", "Serbia and Montenegro": "UEFA",
    "Yugoslavia": "UEFA", "Switzerland": "UEFA", "Austria": "UEFA", "Poland": "UEFA",
    "Ukraine": "UEFA", "Russia": "UEFA", "Czech Republic": "UEFA", "Czechia": "UEFA",
    "Slovakia": "UEFA", "Slovenia": "UEFA", "Romania": "UEFA", "Bulgaria": "UEFA",
    "Hungary": "UEFA", "Greece": "UEFA", "Turkey": "UEFA", "Denmark": "UEFA",
    "Sweden": "UEFA", "Norway": "UEFA", "Finland": "UEFA", "Iceland": "UEFA",
    "Bosnia and Herzegovina": "UEFA", "North Macedonia": "UEFA", "Albania": "UEFA",
    # CONMEBOL
    "Brazil": "CONMEBOL", "Argentina": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Chile": "CONMEBOL", "Colombia": "CONMEBOL", "Peru": "CONMEBOL",
    "Ecuador": "CONMEBOL", "Paraguay": "CONMEBOL", "Bolivia": "CONMEBOL",
    "Venezuela": "CONMEBOL",
    # CONCACAF
    "United States": "CONCACAF", "Mexico": "CONCACAF", "Canada": "CONCACAF",
    "Costa Rica": "CONCACAF", "Honduras": "CONCACAF", "Panama": "CONCACAF",
    "Jamaica": "CONCACAF", "Trinidad and Tobago": "CONCACAF", "Haiti": "CONCACAF",
    "Curaçao": "CONCACAF", "El Salvador": "CONCACAF", "Cuba": "CONCACAF",
    # CAF
    "Nigeria": "CAF", "Ghana": "CAF", "Senegal": "CAF", "Cameroon": "CAF",
    "Ivory Coast": "CAF", "Morocco": "CAF", "Tunisia": "CAF", "Algeria": "CAF",
    "Egypt": "CAF", "South Africa": "CAF", "Togo": "CAF", "Angola": "CAF",
    "DR Congo": "CAF", "Cape Verde": "CAF", "Mali": "CAF", "Burkina Faso": "CAF",
    "Zambia": "CAF",
    # AFC (Australia joined AFC in 2006; treated as AFC throughout for simplicity)
    "Japan": "AFC", "South Korea": "AFC", "North Korea": "AFC", "Saudi Arabia": "AFC",
    "Iran": "AFC", "Iraq": "AFC", "Qatar": "AFC", "United Arab Emirates": "AFC",
    "China": "AFC", "China PR": "AFC", "Australia": "AFC", "Jordan": "AFC",
    "Uzbekistan": "AFC", "Kuwait": "AFC", "Indonesia": "AFC", "Thailand": "AFC",
    # OFC
    "New Zealand": "OFC",
}

# Football name -> ISO3 for the World Bank API. UK constituents share GBR data.
# Only mismatches / non-obvious cases are listed; see iso3() for the fallback.
_ISO3_OVERRIDES: dict[str, str] = {
    "England": "GBR", "Scotland": "GBR", "Wales": "GBR", "Northern Ireland": "GBR",
    "Republic of Ireland": "IRL", "Ireland": "IRL",
    "Germany": "DEU", "Netherlands": "NLD", "Switzerland": "CHE", "Austria": "AUT",
    "Portugal": "PRT", "Denmark": "DNK", "Sweden": "SWE", "Norway": "NOR",
    "Poland": "POL", "Croatia": "HRV", "Serbia": "SRB", "Serbia and Montenegro": "SRB",
    "Slovenia": "SVN", "Slovakia": "SVK", "Czech Republic": "CZE", "Czechia": "CZE",
    "Romania": "ROU", "Bulgaria": "BGR", "Greece": "GRC", "Turkey": "TUR",
    "Ukraine": "UKR", "Russia": "RUS", "Iceland": "ISL", "Finland": "FIN",
    "Hungary": "HUN", "Bosnia and Herzegovina": "BIH", "North Macedonia": "MKD",
    "Albania": "ALB", "France": "FRA", "Italy": "ITA", "Spain": "ESP",
    "Belgium": "BEL", "Yugoslavia": "SRB",  # WB has no Yugoslavia; nearest successor
    "Brazil": "BRA", "Argentina": "ARG", "Uruguay": "URY", "Chile": "CHL",
    "Colombia": "COL", "Peru": "PER", "Ecuador": "ECU", "Paraguay": "PRY",
    "Bolivia": "BOL", "Venezuela": "VEN",
    "United States": "USA", "Mexico": "MEX", "Canada": "CAN", "Costa Rica": "CRI",
    "Honduras": "HND", "Panama": "PAN", "Jamaica": "JAM",
    "Trinidad and Tobago": "TTO", "Haiti": "HTI", "Curaçao": "CUW",
    "El Salvador": "SLV", "Cuba": "CUB",
    "Nigeria": "NGA", "Ghana": "GHA", "Senegal": "SEN", "Cameroon": "CMR",
    "Ivory Coast": "CIV", "Morocco": "MAR", "Tunisia": "TUN", "Algeria": "DZA",
    "Egypt": "EGY", "South Africa": "ZAF", "Togo": "TGO", "Angola": "AGO",
    "DR Congo": "COD", "Cape Verde": "CPV", "Mali": "MLI", "Burkina Faso": "BFA",
    "Zambia": "ZMB",
    "Japan": "JPN", "South Korea": "KOR", "North Korea": "PRK",
    "Saudi Arabia": "SAU", "Iran": "IRN", "Iraq": "IRQ", "Qatar": "QAT",
    "United Arab Emirates": "ARE", "China": "CHN", "China PR": "CHN",
    "Australia": "AUS", "Jordan": "JOR", "Uzbekistan": "UZB", "Kuwait": "KWT",
    "Indonesia": "IDN", "Thailand": "THA",
    "New Zealand": "NZL",
}


def confederation(team: str) -> str:
    """
    Return the confederation for a team, or "OTHER" if unmapped.

    Args:
        team: Team name in results-dataset spelling.

    Returns:
        str: Confederation code (UEFA, CONMEBOL, CONCACAF, ...) or "OTHER".
    """
    return CONFEDERATIONS.get(team, "OTHER")


def iso3(team: str) -> str | None:
    """
    Return the ISO3 code for a team's World Bank data, or None if unmapped.

    UK constituents share GBR; unmapped teams return None so callers can fail soft and
    log the gap rather than crash.

    Args:
        team: Team name in results-dataset spelling.

    Returns:
        str | None: ISO3 country code, or None if the team is not mapped.
    """
    return _ISO3_OVERRIDES.get(team)


def wc_start(year: int) -> str:
    """
    Return the start date of a World Cup edition (its feature freeze date).

    Args:
        year: World Cup edition year (a key in WORLD_CUPS).

    Returns:
        str: The tournament start date as an ISO "YYYY-MM-DD" string.
    """
    return WORLD_CUPS[year]["start"]


def is_host(team: str, year: int) -> bool:
    """
    Return whether a team hosted the given World Cup edition.

    Args:
        team: Team name in results-dataset spelling.
        year: World Cup edition year (a key in WORLD_CUPS).

    Returns:
        bool: True if the team is a host of that edition, else False.
    """
    return team in WORLD_CUPS[year]["hosts"]
