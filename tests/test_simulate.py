"""Tests for predictor/simulate.py."""

import numpy as np
import pytest

from predictor import simulate, table
from tests.conftest import TEAMS, true_model


def half_season(league):
    """The first half of the last season: 15 of 30 matches played."""
    return league[league["season"] == 2002].iloc[:15]


def test_every_simulation_has_consistent_points(league):
    n = len(TEAMS)
    home_goals, away_goals = simulate.sample_remaining(
        true_model(), half_season(league), TEAMS, 500, np.random.default_rng(0))
    everyone = ~np.eye(n, dtype=bool)
    final = table.tally(home_goals, away_goals, everyone)

    assert (final["played"] == 2 * (n - 1)).all()
    # A decided match gives 3 points in total, a draw 2.
    draws = ((home_goals == away_goals) & everyone).sum(axis=(1, 2))
    assert (final["points"].sum(axis=1) == 3 * n * (n - 1) - draws).all()
    assert (final["gd"].sum(axis=1) == 0).all()


def test_played_matches_are_not_resampled(league):
    known = half_season(league)
    home_goals, away_goals = simulate.sample_remaining(
        true_model(), known, TEAMS, 50, np.random.default_rng(0))
    for match in known.itertuples():
        i, j = TEAMS.index(match.home), TEAMS.index(match.away)
        assert (home_goals[:, i, j] == match.hg).all()
        assert (away_goals[:, i, j] == match.ag).all()


def test_remaining_matches_follow_the_model(league):
    # Ada (strongest) at home to Fer (weakest) is not among the played matches.
    known = half_season(league)
    known = known[~((known["home"] == "Ada") & (known["away"] == "Fer"))]
    home_goals, away_goals = simulate.sample_remaining(
        true_model(), known, TEAMS, 4000, np.random.default_rng(0))
    i, j = TEAMS.index("Ada"), TEAMS.index("Fer")
    lam, mu = true_model().expected_goals("Ada", "Fer")
    assert home_goals[:, i, j].mean() == pytest.approx(lam, abs=0.15)
    assert away_goals[:, i, j].mean() == pytest.approx(mu, abs=0.1)


def test_summary_probabilities_are_consistent(league):
    summary, position_probs = simulate.simulate_season(
        true_model(), half_season(league), TEAMS, n_sims=500, relegation=2)
    assert np.allclose(position_probs.sum(axis=1), 1)  # each club finishes somewhere
    assert np.allclose(position_probs.sum(axis=0), 1)  # each place is taken by one club
    assert np.isclose(summary["champion"].sum(), 1)
    assert np.isclose(summary["top2"].sum(), 2)
    assert np.isclose(summary["relegation"].sum(), 2)
    assert (summary["points_p5"] <= summary["exp_points"]).all()
    assert (summary["exp_points"] >= summary["points_now"]).all()


def test_same_seed_gives_same_result(league):
    first, _ = simulate.simulate_season(true_model(), half_season(league), TEAMS, 200)
    second, _ = simulate.simulate_season(true_model(), half_season(league), TEAMS, 200)
    assert first.equals(second)
