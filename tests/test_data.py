"""Tests for predictor/data.py."""

from datetime import date

import pandas as pd
import pytest

from predictor import data


def test_season_code_and_current_season():
    assert data.season_code(2015) == "1516"
    assert data.season_code(2099) == "9900"
    assert data.current_season(date(2026, 6, 30)) == 2025
    assert data.current_season(date(2026, 7, 1)) == 2026


def test_clean_name_maps_spellings_to_one_name():
    assert data.clean_name("Buyuksehyr") == "Başakşehir"
    assert data.clean_name("İstanbul Başakşehir") == "Başakşehir"
    assert data.clean_name("Galatasaray") == "Galatasaray"


def test_remaining_fixtures_count():
    teams = ["A", "B", "C", "D"]  # n = 4, so 4 * 3 = 12 fixtures in total
    played = pd.DataFrame({"home": ["A", "B", "C"], "away": ["B", "A", "D"]})
    remaining = data.remaining_fixtures(played, teams)
    assert len(remaining) == 4 * 3 - 3
    assert ("A", "B") not in set(zip(remaining["home"], remaining["away"]))
    assert ("A", "C") in set(zip(remaining["home"], remaining["away"]))


def test_remaining_fixtures_empty_when_season_is_complete(league):
    season = league[league["season"] == 2000]
    teams = sorted(set(season["home"]))
    assert len(data.remaining_fixtures(season, teams)) == 0


def test_find_team_is_fuzzy():
    teams = ["Fenerbahçe", "Galatasaray", "Gaziantep FK", "Gaziantepspor", "İstanbulspor"]
    assert data.find_team("fener", teams) == "Fenerbahçe"
    assert data.find_team("ISTANBUL", teams) == "İstanbulspor"
    assert data.find_team("galatasary", teams) == "Galatasaray"  # typo
    with pytest.raises(ValueError):
        data.find_team("gaziantep", teams)  # two clubs match
    with pytest.raises(ValueError):
        data.find_team("xyz", teams)


def test_awarded_matches_are_marked():
    matches = pd.DataFrame({
        "season": [2022, 2022, 2022],
        "date": pd.to_datetime(["2023-01-10", "2023-03-05", "2023-03-05"]),
        "home": ["Hatayspor", "Hatayspor", "Galatasaray"],
        "away": ["Konyaspor", "Konyaspor", "Kasımpaşa"],
    })
    assert list(data.mark_awarded(matches)["awarded"]) == [False, True, False]
