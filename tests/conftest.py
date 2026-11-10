"""Shared test data: a small made-up league, so tests do not depend on real results."""

import numpy as np
import pandas as pd
import pytest

from predictor.model import DixonColes

TEAMS = ["Ada", "Bor", "Cem", "Dag", "Ege", "Fer"]


def true_model():
    """A model with hand-picked parameters, used to generate synthetic results."""
    model = DixonColes()
    model.attack = dict(zip(TEAMS, [0.5, 0.3, 0.1, -0.1, -0.3, -0.5]))
    model.defence = dict(zip(TEAMS, [-0.4, -0.2, 0.0, 0.0, 0.2, 0.4]))
    model.intercept, model.home_adv, model.rho = 0.1, 0.3, -0.1
    return model


def play_league(n_seasons, seed=0):
    """Sample `n_seasons` double round-robins from `true_model`, one round per week."""
    rng = np.random.default_rng(seed)
    model = true_model()
    rows = []
    for season in range(n_seasons):
        day = pd.Timestamp(2000 + season, 8, 1)
        pairs = [(home, away) for home in TEAMS for away in TEAMS if home != away]
        rng.shuffle(pairs)
        for number, (home, away) in enumerate(pairs):
            cell = rng.choice(121, p=model.score_matrix(home, away).ravel())
            rows.append({
                "season": 2000 + season, "date": day + pd.Timedelta(days=7 * (number // 3)),
                "home": home, "away": away, "hg": cell // 11, "ag": cell % 11,
                "odds_h": 2.2, "odds_d": 3.3, "odds_a": 3.4, "awarded": False,
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def league():
    """Three seasons of a six-club league."""
    return play_league(3)
