"""Download, clean and store Süper Lig results.

Main source: football-data.co.uk (results and bookmaker odds, one CSV per season).
Fallback source: openfootball (results only, one JSON per season, many seasons missing).

A season is identified by its start year: 2015 means 2015/16.
"""

import difflib
import json
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
MATCHES_CSV = DATA_DIR / "matches.csv"

FOOTBALL_DATA_URL = "https://www.football-data.co.uk/mmz4281/{code}/T1.csv"
OPENFOOTBALL_URL = (
    "https://raw.githubusercontent.com/openfootball/football.json/master/{name}/tr.1.json"
)

COLUMNS = ["season", "date", "home", "away", "hg", "ag", "odds_h", "odds_d", "odds_a",
           "awarded"]

# Clubs that withdrew during a season: season -> (clubs, date of withdrawal).
# Their later matches were never played and were recorded as 3-0 defeats.
# They count for the league table but say nothing about playing strength.
# 2022/23: Hatayspor and Gaziantep FK withdrew after the earthquake of 6 Feb 2023.
WITHDRAWALS = {2022: (["Hatayspor", "Gaziantep FK"], "2023-02-06")}

# Odds column sets in order of preference: market average (new name), market
# average (old "BetBrain" name), Bet365, Pinnacle.
ODDS_COLUMN_SETS = [
    ("AvgH", "AvgD", "AvgA"),
    ("BbAvH", "BbAvD", "BbAvA"),
    ("B365H", "B365D", "B365A"),
    ("PSH", "PSD", "PSA"),
]

# Every spelling used by the two sources -> the one name we use everywhere.
# Names that are already fine (Galatasaray, Trabzonspor, ...) are not listed.
ALIASES = {
    # football-data.co.uk spellings
    "Ad. Demirspor": "Adana Demirspor",
    "Akhisar Belediyespor": "Akhisarspor",
    "Ankaragucu": "Ankaragücü",
    "Besiktas": "Beşiktaş",
    "Bodrumspor": "Bodrum FK",
    "Buyuksehyr": "Başakşehir",
    "Corum": "Çorum FK",
    "Erzurum BB": "Erzurumspor",
    "Eskisehirspor": "Eskişehirspor",
    "Eyupspor": "Eyüpspor",
    "Fenerbahce": "Fenerbahçe",
    "Gaziantep": "Gaziantep FK",  # not the same club as the older Gaziantepspor
    "Genclerbirligi": "Gençlerbirliği",
    "Goztep": "Göztepe",
    "Istanbulspor": "İstanbulspor",
    "Karabukspor": "Karabükspor",
    "Karagumruk": "Karagümrük",
    "Kasimpasa": "Kasımpaşa",
    "Mersin Idman Yurdu": "Mersin İdman Yurdu",
    "Osmanlispor": "Osmanlıspor",
    "Umraniyespor": "Ümraniyespor",
    # openfootball spellings
    "Atiker Konyaspor": "Konyaspor",
    "BB Erzurumspor": "Erzurumspor",
    "Basaksehir": "Başakşehir",
    "Çaykur Rizespor": "Rizespor",
    "Caykur Rizespor": "Rizespor",
    "Fatih Karagümrük": "Karagümrük",
    "Gazişehir Gaziantep FK": "Gaziantep FK",
    "İstanbul Başakşehir": "Başakşehir",
    "Kasımpaşa SK": "Kasımpaşa",
    "MKE Ankaragücü": "Ankaragücü",
}


def current_season(today=None):
    """Return the start year of the season running on `today` (seasons start in July)."""
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def season_code(season):
    """Return the football-data.co.uk code of a season, e.g. 2015 -> '1516'."""
    return f"{season % 100:02d}{(season + 1) % 100:02d}"


def season_label(season):
    """Return a readable season name, e.g. 2015 -> '2015/16'."""
    return f"{season}/{(season + 1) % 100:02d}"


def clean_name(name):
    """Return the single spelling we use for a club."""
    name = str(name).strip()
    return ALIASES.get(name, name)


def simplify(name):
    """Lower-case a name and replace Turkish letters, so 'Fenerbahçe' -> 'fenerbahce'."""
    # "İ".lower() gives "i" plus a combining dot (U+0307), which we delete.
    return name.lower().translate(str.maketrans("çğıöşü", "cgiosu", "\u0307"))


def find_team(query, teams):
    """Return the club in `teams` that best matches what the user typed.

    Tries, in order: exact name, start of the name, part of the name, closest
    spelling. Raises ValueError if nothing fits or the query is ambiguous.
    """
    wanted = simplify(query.strip())
    simple = {team: simplify(team) for team in teams}
    exact = [team for team in teams if simple[team] == wanted]
    starts = [team for team in teams if simple[team].startswith(wanted)]
    contains = [team for team in teams if wanted in simple[team]]
    close = difflib.get_close_matches(wanted, simple.values(), n=1, cutoff=0.6)
    similar = [team for team in teams if simple[team] in close]
    for candidates in (exact, starts, contains, similar):
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError(f"'{query}' is ambiguous: {', '.join(candidates)}")
    raise ValueError(f"no club matches '{query}'")


def mark_awarded(matches):
    """Add the boolean column `awarded`: True for matches decided without being played."""
    matches["awarded"] = False
    for season, (clubs, withdrawal_date) in WITHDRAWALS.items():
        involved = matches["home"].isin(clubs) | matches["away"].isin(clubs)
        after = matches["date"] > pd.Timestamp(withdrawal_date)
        matches.loc[(matches["season"] == season) & involved & after, "awarded"] = True
    return matches


def download(url, path, use_cache):
    """Download `url` to `path`. If `use_cache` is set and the file exists, do nothing."""
    if use_cache and path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        path.write_bytes(response.read())


def pick_odds(raw):
    """Return (odds table, names of the odds columns found in the file).

    For each match we take the first column set in ODDS_COLUMN_SETS that has
    all three prices, so a gap in the market average is filled by Bet365 etc.
    """
    odds = pd.DataFrame(index=raw.index, columns=["odds_h", "odds_d", "odds_a"], dtype=float)
    found = []
    for columns in ODDS_COLUMN_SETS:
        if not set(columns) <= set(raw.columns):
            continue
        found.append(columns[0])
        candidate = raw[list(columns)].apply(pd.to_numeric, errors="coerce")
        candidate.columns = odds.columns
        complete = candidate.notna().all(axis=1)
        missing = odds.isna().any(axis=1)
        odds[missing & complete] = candidate[missing & complete]
    return odds, found


def parse_football_data(path, season):
    """Read one football-data.co.uk CSV into our column layout."""
    raw = pd.read_csv(path, encoding="latin-1")
    raw = raw.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    odds, found = pick_odds(raw)
    matches = pd.DataFrame(
        {
            "season": season,
            # Old files use dd/mm/yy, newer ones dd/mm/yyyy.
            "date": pd.to_datetime(raw["Date"], dayfirst=True, format="mixed"),
            "home": raw["HomeTeam"].map(clean_name),
            "away": raw["AwayTeam"].map(clean_name),
            "hg": raw["FTHG"].astype(int),
            "ag": raw["FTAG"].astype(int),
        }
    )
    matches = pd.concat([matches, odds], axis=1)
    with_odds = int(matches["odds_h"].notna().sum())
    print(f"  football-data: {len(matches)} matches, odds columns {found}, "
          f"{with_odds} matches with odds")
    return matches


def parse_openfootball(path, season):
    """Read one openfootball JSON file into our column layout (no odds)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for match in data["matches"]:
        score = match.get("score")
        # Only a dict with "ft" (full time) is a played match.
        if not isinstance(score, dict) or "ft" not in score:
            continue
        rows.append(
            {
                "season": season,
                "date": pd.Timestamp(match["date"]),
                "home": clean_name(match["team1"]),
                "away": clean_name(match["team2"]),
                "hg": int(score["ft"][0]),
                "ag": int(score["ft"][1]),
                "odds_h": float("nan"),
                "odds_d": float("nan"),
                "odds_a": float("nan"),
            }
        )
    print(f"  openfootball: {len(rows)} played matches (no odds)")
    return pd.DataFrame(rows, columns=COLUMNS[:-1])


def fetch_season(season, is_finished):
    """Download and parse one season. Returns an empty table if no source has it.

    A finished season never changes, so its raw file is cached. The current
    season is downloaded again every time.
    """
    csv_path = RAW_DIR / f"T1_{season_code(season)}.csv"
    try:
        download(FOOTBALL_DATA_URL.format(code=season_code(season)), csv_path, is_finished)
        return parse_football_data(csv_path, season)
    except Exception as error:
        print(f"  football-data failed ({error}); trying openfootball")

    json_path = RAW_DIR / f"openfootball_{season}.json"
    name = f"{season}-{(season + 1) % 100:02d}"
    try:
        download(OPENFOOTBALL_URL.format(name=name), json_path, is_finished)
        return parse_openfootball(json_path, season)
    except Exception as error:
        print(f"  openfootball failed ({error}); season skipped")
        return pd.DataFrame(columns=COLUMNS[:-1])


def fetch_all(first=2015, last=None):
    """Download seasons `first`..`last`, save data/matches.csv and return the table."""
    current = current_season()
    last = current if last is None else last
    seasons = []
    for season in range(first, last + 1):
        print(f"{season_label(season)}:")
        seasons.append(fetch_season(season, is_finished=season < current))
    matches = pd.concat([s for s in seasons if len(s)], ignore_index=True)
    matches = matches.sort_values(["date", "home"], kind="stable").reset_index(drop=True)
    matches = mark_awarded(matches)
    MATCHES_CSV.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(MATCHES_CSV, index=False, encoding="utf-8")
    return matches


def load_matches(path=MATCHES_CSV):
    """Load the processed results table saved by `fetch_all`."""
    return pd.read_csv(path, parse_dates=["date"], encoding="utf-8")


def season_teams(matches, season):
    """Return the sorted list of clubs that appear in a season."""
    rows = matches[matches["season"] == season]
    return sorted(set(rows["home"]) | set(rows["away"]))


def remaining_fixtures(season_matches, teams):
    """Return every (home, away) pairing of a double round-robin not yet played.

    In a double round-robin every club hosts every other club exactly once, so
    there are n * (n - 1) fixtures in total.
    """
    played = set(zip(season_matches["home"], season_matches["away"]))
    fixtures = [
        (home, away)
        for home in teams
        for away in teams
        if home != away and (home, away) not in played
    ]
    return pd.DataFrame(fixtures, columns=["home", "away"])


def print_summary(matches):
    """Print matches per season, the date range and the club names per season."""
    print(f"\n{len(matches)} matches from {matches['date'].min():%Y-%m-%d} "
          f"to {matches['date'].max():%Y-%m-%d}\n")
    print(f"{'season':<9}{'matches':>8}{'clubs':>7}{'with odds':>11}{'awarded':>9}   first .. last")
    for season, rows in matches.groupby("season"):
        clubs = len(season_teams(matches, season))
        print(f"{season_label(season):<9}{len(rows):>8}{clubs:>7}"
              f"{int(rows['odds_h'].notna().sum()):>11}{int(rows['awarded'].sum()):>9}   "
              f"{rows['date'].min():%Y-%m-%d} .. {rows['date'].max():%Y-%m-%d}")
    print("\nClub names per season:")
    for season in sorted(matches["season"].unique()):
        print(f"{season_label(season)}: {', '.join(season_teams(matches, season))}")
