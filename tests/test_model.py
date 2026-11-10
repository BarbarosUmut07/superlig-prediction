"""Tests for predictor/model.py."""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime

from predictor.model import DixonColes, negative_log_likelihood, tau
from tests.conftest import TEAMS, play_league, true_model


def test_score_matrix_is_a_probability_distribution():
    model = true_model()
    for home, away in [("Ada", "Fer"), ("Fer", "Ada"), ("Cem", "Dag")]:
        matrix = model.score_matrix(home, away)
        assert matrix.shape == (11, 11)
        assert matrix.min() >= 0
        assert matrix.sum() == pytest.approx(1.0)


def test_outcome_probs_sum_to_one_and_favour_the_stronger_club():
    home_win, draw, away_win = true_model().outcome_probs("Ada", "Fer")
    assert home_win + draw + away_win == pytest.approx(1.0)
    assert home_win > draw > away_win


def test_tau_only_changes_the_four_low_scores():
    lam, mu, rho = 1.4, 1.1, -0.1
    for x in range(6):
        for y in range(6):
            value = float(tau(np.array(x), np.array(y), np.array(lam), np.array(mu), rho))
            if (x, y) == (0, 0):
                assert value == pytest.approx(1 - lam * mu * rho)
            elif (x, y) == (0, 1):
                assert value == pytest.approx(1 + lam * rho)
            elif (x, y) == (1, 0):
                assert value == pytest.approx(1 + mu * rho)
            elif (x, y) == (1, 1):
                assert value == pytest.approx(1 - rho)
            else:
                assert value == 1.0


def test_gradient_matches_numerical_gradient():
    rng = np.random.default_rng(0)
    n, k = 5, 200
    home = rng.integers(0, n, k)
    away = (home + rng.integers(1, n, k)) % n
    args = (home, away, rng.poisson(1.4, k), rng.poisson(1.1, k), rng.random(k), n, 0.5)
    params = rng.normal(0, 0.2, 2 * n + 3)
    params[-1] = 0.1  # rho
    numerical = approx_fprime(params, lambda p: negative_log_likelihood(p, *args)[0], 1e-7)
    assert negative_log_likelihood(params, *args)[1] == pytest.approx(numerical, abs=1e-3)


def test_model_recovers_known_strengths():
    matches = play_league(40, seed=1)  # 1200 matches from known parameters
    truth = true_model()
    model = DixonColes(xi=0.0, ridge=0.01).fit(matches)  # no decay, almost no shrinkage
    assert model.home_adv == pytest.approx(truth.home_adv, abs=0.1)
    assert model.rho == pytest.approx(truth.rho, abs=0.1)
    for team in TEAMS:
        assert model.attack[team] == pytest.approx(truth.attack[team], abs=0.15)
        assert model.defence[team] == pytest.approx(truth.defence[team], abs=0.15)
    assert list(model.ratings()["team"]) == TEAMS  # strongest to weakest


def test_fit_ignores_matches_on_or_after_as_of(league):
    as_of = league.loc[league["season"] == 2001, "date"].iloc[12]  # a real match day
    changed = league.copy()
    changed.loc[changed["date"] >= as_of, ["hg", "ag"]] = [9, 0]
    original = DixonColes().fit(league, as_of)
    other = DixonColes().fit(changed, as_of)
    assert original.attack == pytest.approx(other.attack)
    assert original.home_adv == pytest.approx(other.home_adv)


def test_fit_ignores_awarded_matches(league):
    extra = league.iloc[:5].copy()
    extra["hg"], extra["ag"], extra["awarded"] = 3, 0, True
    with_awarded = pd.concat([league, extra], ignore_index=True)
    assert DixonColes().fit(with_awarded).attack == pytest.approx(DixonColes().fit(league).attack)


def test_unknown_club_is_slightly_below_average():
    model = true_model()
    lam_new, mu_new = model.expected_goals("New FC", "Cem")
    lam_avg, mu_avg = model.expected_goals("Cem", "Cem")  # Cem is an average club
    assert lam_new < lam_avg
    assert mu_new > mu_avg


def test_ridge_pulls_ratings_towards_zero(league):
    loose = DixonColes(ridge=0.01).fit(league)
    tight = DixonColes(ridge=50.0).fit(league)
    assert max(map(abs, tight.attack.values())) < max(map(abs, loose.attack.values()))
