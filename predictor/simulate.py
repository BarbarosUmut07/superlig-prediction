"""Monte Carlo simulation of the rest of a season.

Every remaining fixture gets one sampled score per simulation, drawn from the
model's score matrix. Each simulation therefore produces one complete final
table, and counting over all simulations gives the chances of every outcome.
"""

import numpy as np
import pandas as pd

from predictor import data, table
from predictor.model import MAX_GOALS


def sample_remaining(model, season_matches, teams, n_sims, rng):
    """Play out the rest of the season `n_sims` times.

    Returns (home_goals, away_goals) with shape (n_sims, n, n): the matches
    already played are the same in every simulation, the remaining ones are
    sampled from the model.
    """
    home_goals, away_goals, _ = table.results_matrices(season_matches, teams)
    home_goals = np.repeat(home_goals[None, :, :], n_sims, axis=0)
    away_goals = np.repeat(away_goals[None, :, :], n_sims, axis=0)

    index = {team: i for i, team in enumerate(teams)}
    size = MAX_GOALS + 1
    for home, away in data.remaining_fixtures(season_matches, teams).itertuples(index=False):
        probabilities = model.score_matrix(home, away).ravel()
        # One draw picks a cell of the flattened 11x11 score matrix;
        # cell = home goals * 11 + away goals. All simulations at once.
        cells = rng.choice(size * size, size=n_sims, p=probabilities)
        home_goals[:, index[home], index[away]] = cells // size
        away_goals[:, index[home], index[away]] = cells % size
    return home_goals, away_goals


def simulate_season(model, season_matches, teams, n_sims=10_000, relegation=3,
                    deductions=None, seed=42):
    """Simulate the rest of a season and summarise the results per club.

    Returns (summary, position_probs):
    summary         one row per club: points now, expected points, 5th-95th
                    percentile range, expected position and the chances of
                    champion, top 2, top 4 and relegation,
    position_probs  clubs x final positions table of probabilities.
    """
    rng = np.random.default_rng(seed)
    n = len(teams)
    deductions = deductions or {}
    deducted = np.array([deductions.get(team, 0) for team in teams])

    now = table.league_table(season_matches, teams, deductions).set_index("team")

    home_goals, away_goals = sample_remaining(model, season_matches, teams, n_sims, rng)
    everyone = ~np.eye(n, dtype=bool)  # at the end every pair has played
    final = table.tally(home_goals, away_goals, everyone)
    points = final["points"] - deducted
    order = table.rank_many(points, final["gd"], final["gf"], home_goals, away_goals)

    # order[s, place] = club  ->  place_of[s, club] = place (0 = champion)
    place_of = np.empty_like(order)
    np.put_along_axis(place_of, order, np.arange(n)[None, :], axis=1)

    summary = pd.DataFrame({
        "team": teams,
        "played": now.loc[teams, "played"].to_numpy(),
        "points_now": now.loc[teams, "points"].to_numpy(),
        "exp_points": points.mean(axis=0),
        "points_p5": np.percentile(points, 5, axis=0),
        "points_p95": np.percentile(points, 95, axis=0),
        "exp_position": place_of.mean(axis=0) + 1,
        "champion": (place_of == 0).mean(axis=0),
        "top2": (place_of < 2).mean(axis=0),
        "top4": (place_of < 4).mean(axis=0),
        "relegation": (place_of >= n - relegation).mean(axis=0),
    })
    summary = summary.sort_values(["exp_points", "exp_position"], ascending=[False, True])
    summary = summary.reset_index(drop=True)

    position_probs = pd.DataFrame(
        [(place_of == place).mean(axis=0) for place in range(n)],
        index=range(1, n + 1),
        columns=teams,
    ).T.loc[summary["team"]]
    return summary, position_probs
