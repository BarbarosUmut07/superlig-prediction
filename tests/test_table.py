"""Tests for predictor/table.py."""

import numpy as np
import pandas as pd

from predictor import table


def mini_league(rows):
    """Build a matches table from (home, away, home goals, away goals) tuples."""
    return pd.DataFrame(rows, columns=["home", "away", "hg", "ag"])


# A and B both finish on 7 points. A has the better goal difference (+5 vs +1),
# but B took 4 points from the two matches against A.
FINISHED = [
    ("A", "B", 0, 1), ("B", "A", 1, 1),
    ("A", "C", 5, 0), ("C", "A", 0, 1),
    ("B", "C", 1, 0), ("C", "B", 2, 1),
]


def test_table_counts_results():
    standings = table.league_table(mini_league(FINISHED)).set_index("team")
    row = standings.loc["A"]
    assert (row.played, row.won, row.drawn, row.lost) == (4, 2, 1, 1)
    assert (row.gf, row.ga, row.gd, row.points) == (7, 2, 5, 7)
    assert standings["points"].sum() == 3 * 5 + 2 * 1  # five decided matches, one draw


def test_head_to_head_changes_the_order():
    standings = table.league_table(mini_league(FINISHED))
    assert list(standings["team"]) == ["B", "A", "C"]
    # Without the rule, goal difference would put A first.
    home_goals, away_goals, played = table.results_matrices(mini_league(FINISHED), ["A", "B", "C"])
    stats = table.tally(home_goals, away_goals, played)
    assert list(table.simple_order(stats["points"], stats["gd"], stats["gf"])) == [0, 1, 2]


def test_head_to_head_waits_until_both_matches_are_played():
    unfinished = [row for row in FINISHED if row[:2] != ("B", "A")]
    standings = table.league_table(mini_league(unfinished), teams=["A", "B", "C"])
    assert standings.set_index("team").loc[["A", "B"], "points"].tolist() == [6, 6]
    assert list(standings["team"]) == ["A", "B", "C"]  # goal difference decides for now


def test_three_way_tie_uses_only_matches_between_the_tied_clubs():
    # A, B and C each beat D twice and are level on 12 points. Among themselves
    # they all took 6 points, so the mini-table goal difference decides.
    rows = [
        ("A", "B", 3, 0), ("B", "A", 1, 0),
        ("B", "C", 2, 0), ("C", "B", 1, 0),
        ("C", "A", 1, 0), ("A", "C", 1, 0),
        ("A", "D", 1, 0), ("D", "A", 0, 1), ("B", "D", 1, 0), ("D", "B", 0, 1),
        ("C", "D", 9, 0), ("D", "C", 0, 9),
    ]
    standings = table.league_table(mini_league(rows))
    assert standings["points"].tolist() == [12, 12, 12, 0]
    # Overall goal difference would put C first (+17), but C's big wins against D
    # do not count in the mini-table: A +2, B -1 (3 goals scored), C -1 (2 goals).
    assert list(standings["team"]) == ["A", "B", "C", "D"]


def test_point_deductions_are_applied():
    standings = table.league_table(mini_league(FINISHED), deductions={"B": 3})
    assert list(standings["team"]) == ["A", "B", "C"]
    assert standings.set_index("team").loc["B", "points"] == 4


def test_load_deductions_reads_one_season(tmp_path):
    path = tmp_path / "deductions.csv"
    path.write_text("season,team,points\n2023,X,3\n2024,Y,6\n", encoding="utf-8")
    assert table.load_deductions(2023, path) == {"X": 3}
    assert table.load_deductions(2023, tmp_path / "missing.csv") == {}


def test_rank_many_agrees_with_rank():
    rng = np.random.default_rng(3)
    n = 6
    home_goals = rng.poisson(1.5, (300, n, n))
    away_goals = rng.poisson(1.2, (300, n, n))
    stats = table.tally(home_goals, away_goals, ~np.eye(n, dtype=bool))
    many = table.rank_many(stats["points"], stats["gd"], stats["gf"], home_goals, away_goals)
    for s in range(300):
        one = table.rank(stats["points"][s], stats["gd"][s], stats["gf"][s],
                         home_goals[s], away_goals[s])
        assert list(many[s]) == list(one)
        assert sorted(many[s]) == list(range(n))  # every club appears exactly once
