"""League table and Süper Lig tie-break rules.

Results are stored in three n x n arrays, one row and column per club:

    home_goals[i, j]  goals of club i at home against club j
    away_goals[i, j]  goals of club j away at club i
    played[i, j]      True if club i has already hosted club j

In a double round-robin every (home, away) pair occurs exactly once, so these
arrays hold a whole season. The same functions also accept arrays with an
extra first axis (one table per simulation), which the simulator uses.

Tie-break for clubs level on points (TFF rule):
    1. points in the matches between the tied clubs,
    2. goal difference in those matches,
    3. goals scored in those matches,
    4. overall goal difference,
    5. overall goals scored.
The head-to-head steps 1-3 apply only once the tied clubs have played all
their matches against each other; until then only steps 4 and 5 are used.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DEDUCTIONS_CSV = Path(__file__).resolve().parent.parent / "data" / "deductions.csv"


def results_matrices(matches, teams):
    """Turn a season's matches into (home_goals, away_goals, played) arrays."""
    index = {team: i for i, team in enumerate(teams)}
    n = len(teams)
    home_goals = np.zeros((n, n), dtype=np.int16)
    away_goals = np.zeros((n, n), dtype=np.int16)
    played = np.zeros((n, n), dtype=bool)
    i = matches["home"].map(index).to_numpy()
    j = matches["away"].map(index).to_numpy()
    home_goals[i, j] = matches["hg"].to_numpy()
    away_goals[i, j] = matches["ag"].to_numpy()
    played[i, j] = True
    return home_goals, away_goals, played


def tally(home_goals, away_goals, played):
    """Count wins, draws, losses, goals and points per club.

    Summing over axis -1 (a row) adds up a club's home matches, summing over
    axis -2 (a column) its away matches. Returns a dict of arrays.
    """
    home_win = played & (home_goals > away_goals)
    draw = played & (home_goals == away_goals)
    away_win = played & (home_goals < away_goals)
    won = home_win.sum(-1) + away_win.sum(-2)
    drawn = draw.sum(-1) + draw.sum(-2)
    lost = away_win.sum(-1) + home_win.sum(-2)
    goals_for = (home_goals * played).sum(-1) + (away_goals * played).sum(-2)
    goals_against = (away_goals * played).sum(-1) + (home_goals * played).sum(-2)
    return {
        "played": won + drawn + lost,
        "won": won,
        "drawn": drawn,
        "lost": lost,
        "gf": goals_for,
        "ga": goals_against,
        "gd": goals_for - goals_against,
        "points": 3 * won + drawn,
    }


def head_to_head_order(group, home_goals, away_goals, gd, gf):
    """Order the clubs in `group` (indices, level on points) by the tie-break rule.

    Builds a mini-table from only the matches between these clubs and sorts
    by its points, goal difference and goals, then by overall goal difference
    and overall goals.
    """
    rows = np.ix_(group, group)  # the k x k block of matches inside the group
    everyone = np.ones((len(group), len(group)), dtype=bool)
    np.fill_diagonal(everyone, False)  # a club does not play itself
    mini = tally(home_goals[rows], away_goals[rows], everyone)
    # lexsort sorts by the LAST key first; minus signs make it descending.
    order = np.lexsort((-gf[group], -gd[group], -mini["gf"], -mini["gd"], -mini["points"]))
    return group[order]


def simple_order(points, gd, gf):
    """Fast path: order clubs by points, then goal difference, then goals scored.

    The three numbers are packed into one sort key, e.g. 71 points, +30 and
    65 goals -> 71_1030_065. Works for one table or for many (last axis = clubs).
    """
    key = points.astype(np.int64) * 10_000_000 + (gd + 1000) * 1000 + gf
    return np.argsort(-key, axis=-1, kind="stable")


def rank(points, gd, gf, home_goals, away_goals, played=None, order=None):
    """Return club indices from first to last place for one table.

    Starts from the fast-path order and re-orders every run of clubs level on
    points with the head-to-head rule, if they have finished their matches
    against each other (`played=None` means the season is complete).
    """
    order = simple_order(points, gd, gf) if order is None else order
    order = order.copy()
    sorted_points = points[order]
    start = 0
    for end in range(1, len(order) + 1):
        if end < len(order) and sorted_points[end] == sorted_points[start]:
            continue  # still inside a run of equal points
        if end - start > 1:
            group = order[start:end]
            rows = np.ix_(group, group)
            needed = (end - start) * (end - start - 1)  # matches inside the group
            if played is None or played[rows].sum() == needed:
                order[start:end] = head_to_head_order(group, home_goals, away_goals, gd, gf)
        start = end
    return order


def rank_many(points, gd, gf, home_goals, away_goals):
    """Rank many finished seasons at once; arrays have a leading simulation axis.

    Returns `order[s, place]` = index of the club finishing in that place.
    The fast path ranks all simulations in one vectorised step. Only the
    simulations in which at least two clubs are level on points go through
    the slower head-to-head rule.
    """
    order = simple_order(points, gd, gf)
    sorted_points = np.take_along_axis(points, order, axis=1)
    has_tie = (sorted_points[:, 1:] == sorted_points[:, :-1]).any(axis=1)
    for s in np.flatnonzero(has_tie):
        order[s] = rank(points[s], gd[s], gf[s], home_goals[s], away_goals[s], order=order[s])
    return order


def load_deductions(season, path=DEDUCTIONS_CSV):
    """Return {club: points deducted} for a season from data/deductions.csv, if it exists."""
    if not Path(path).exists():
        return {}
    table = pd.read_csv(path, encoding="utf-8")
    table = table[table["season"] == season]
    return dict(zip(table["team"], table["points"]))


def league_table(matches, teams=None, deductions=None):
    """Build the ranked league table of one season's matches.

    `deductions` maps a club to the number of points taken away from it.
    """
    if teams is None:
        teams = sorted(set(matches["home"]) | set(matches["away"]))
    deductions = deductions or {}
    home_goals, away_goals, played = results_matrices(matches, teams)
    stats = tally(home_goals, away_goals, played)
    stats["points"] = stats["points"] - np.array([deductions.get(team, 0) for team in teams])
    order = rank(stats["points"], stats["gd"], stats["gf"], home_goals, away_goals, played)

    table = pd.DataFrame({"team": teams, **stats})
    table = table.iloc[order].reset_index(drop=True)
    table.index = table.index + 1  # positions start at 1
    return table
