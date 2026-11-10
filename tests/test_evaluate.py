"""Tests for predictor/evaluate.py."""

import numpy as np
import pandas as pd
import pytest

from predictor import evaluate


def test_rps_by_hand():
    probs = np.array([[0.5, 0.3, 0.2]])
    assert evaluate.rps(probs, np.array([0])) == pytest.approx((0.5 ** 2 + 0.2 ** 2) / 2)
    assert evaluate.rps(probs, np.array([1])) == pytest.approx((0.5 ** 2 + 0.2 ** 2) / 2)
    assert evaluate.rps(probs, np.array([2])) == pytest.approx((0.5 ** 2 + 0.8 ** 2) / 2)


def test_rps_punishes_a_far_miss_more_than_a_near_miss():
    home_favourite = np.array([[0.7, 0.2, 0.1]])
    assert evaluate.rps(home_favourite, np.array([1])) < evaluate.rps(home_favourite, np.array([2]))


def test_other_scores_by_hand():
    probs = np.array([[0.5, 0.3, 0.2]])
    assert evaluate.log_loss(probs, np.array([0])) == pytest.approx(np.log(2))
    assert evaluate.brier(probs, np.array([0])) == pytest.approx(0.25 + 0.09 + 0.04)
    assert evaluate.accuracy(probs, np.array([0])) == 1.0
    assert evaluate.accuracy(probs, np.array([2])) == 0.0


def test_perfect_forecast_scores_zero():
    probs = np.eye(3)
    outcome = np.array([0, 1, 2])
    assert evaluate.rps(probs, outcome) == 0
    assert evaluate.brier(probs, outcome) == 0


def test_bookmaker_probs_remove_the_margin():
    matches = pd.DataFrame({"odds_h": [2.0, 1.5], "odds_d": [3.4, 4.0], "odds_a": [3.6, 6.5]})
    probs = evaluate.bookmaker_probs(matches)
    assert probs.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert probs[0, 0] < 1 / 2.0  # smaller than the raw 1/odds, which includes the margin


def test_backtest_covers_every_match_once(league):
    predictions = evaluate.walk_forward(league, 2002)
    assert len(predictions) == (league["season"] == 2002).sum()
    for key in evaluate.FORECASTERS:
        assert evaluate.probs_of(predictions, key).sum(axis=1) == pytest.approx(1.0)


def test_backtest_never_uses_matches_on_or_after_the_prediction_date(league):
    """Leakage test: rewrite the future and check that today's predictions stay the same."""
    days = sorted(league.loc[league["season"] == 2002, "date"].unique())
    cut = days[len(days) // 2]
    rewritten = league.copy()
    rewritten.loc[rewritten["date"] >= cut, ["hg", "ag"]] = [9, 0]

    columns = ["model_h", "model_d", "model_a", "base_h", "base_d", "base_a"]
    original = evaluate.walk_forward(league, 2002)
    changed = evaluate.walk_forward(rewritten, 2002)

    up_to_cut = original["date"] <= cut  # includes the matches played on `cut` itself
    assert np.allclose(original.loc[up_to_cut, columns], changed.loc[up_to_cut, columns])
    # Sanity check: later predictions do change, so the test can actually fail.
    assert not np.allclose(original.loc[~up_to_cut, columns], changed.loc[~up_to_cut, columns])


def test_calibration_bins_count_every_probability(league):
    predictions = evaluate.walk_forward(league, 2002)
    bins = evaluate.calibration(predictions, "model")
    assert bins["count"].sum() == 3 * len(predictions)
    assert bins["observed"].between(0, 1).all()
