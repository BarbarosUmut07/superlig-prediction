"""Command line for the Süper Lig prediction model.

Usage:
    python predict.py fetch [--first 2015] [--last 2026]
    python predict.py ratings [--as-of YYYY-MM-DD]
    python predict.py match <home> <away> [--as-of YYYY-MM-DD]
"""

import argparse
import sys
from datetime import date

import numpy as np

from predictor import charts, data
from predictor.model import DixonColes


def fit_model(matches, as_of):
    """Fit a Dixon-Coles model on all matches before `as_of` (None = all matches)."""
    model = DixonColes().fit(matches, as_of)
    print(f"Model fitted on matches before {model.as_of:%Y-%m-%d}: "
          f"home advantage {model.home_adv:.3f} (x{np.exp(model.home_adv):.2f} goals), "
          f"rho {model.rho:.3f}\n")
    return model


def season_on(matches, as_of):
    """Return the season in progress on `as_of` (None = the latest season in the data)."""
    if as_of is None:
        return int(matches["season"].max())
    return data.current_season(as_of)


def cmd_fetch(args):
    """Download all seasons, save data/matches.csv and print a summary."""
    matches = data.fetch_all(args.first, args.last)
    data.print_summary(matches)
    print(f"\nSaved {len(matches)} matches to {data.MATCHES_CSV}")


def cmd_ratings(args):
    """Print attack, defence and overall rating of the season's clubs."""
    matches = data.load_matches()
    model = fit_model(matches, args.as_of)
    teams = data.season_teams(matches, season_on(matches, args.as_of))
    table = model.ratings()
    table = table[table["team"].isin(teams)].reset_index(drop=True)
    print(f"{'':>3} {'club':<16}{'attack':>8}{'defence':>9}{'rating':>8}")
    for rank, row in enumerate(table.itertuples(), start=1):
        print(f"{rank:>3} {row.team:<16}{row.attack:>+8.2f}{row.defence:>+9.2f}{row.rating:>+8.2f}")
    missing = sorted(set(teams) - set(table["team"]))
    if missing:
        print(f"\nNo matches yet (treated as slightly below average): {', '.join(missing)}")


def cmd_match(args):
    """Predict one match: expected goals, outcome chances and likely scores."""
    matches = data.load_matches()
    season = season_on(matches, args.as_of)
    all_teams = sorted(set(matches["home"]) | set(matches["away"]))

    def find(query):
        # Prefer this season's clubs, so "gaziantep" means Gaziantep FK today.
        try:
            return data.find_team(query, data.season_teams(matches, season))
        except ValueError:
            return data.find_team(query, all_teams)

    try:
        home, away = find(args.home), find(args.away)
    except ValueError as error:
        sys.exit(f"Error: {error}")
    model = fit_model(matches, args.as_of)

    lam, mu = model.expected_goals(home, away)
    home_win, draw, away_win = model.outcome_probs(home, away)
    print(f"{home} vs {away}")
    print(f"Expected goals: {home} {lam:.2f}, {away} {mu:.2f}")
    print(f"{home} win {home_win:.1%} | draw {draw:.1%} | {away} win {away_win:.1%}")

    scores = model.score_matrix(home, away)
    best = np.argsort(scores, axis=None)[::-1][:5]  # the five largest cells
    print("Most likely scores:")
    for flat_index in best:
        x, y = np.unravel_index(flat_index, scores.shape)
        print(f"  {x}-{y}  {scores[x, y]:.1%}")
    print(f"\nChart saved to {charts.match_chart(model, home, away)}")


def parse_date(text):
    """Turn YYYY-MM-DD from the command line into a date."""
    return date.fromisoformat(text)


def build_parser():
    """Create the argument parser with one sub-command per task."""
    parser = argparse.ArgumentParser(description="Süper Lig prediction model")
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch", help="download results and build data/matches.csv")
    fetch.add_argument("--first", type=int, default=2015, help="first season (start year)")
    fetch.add_argument("--last", type=int, default=None,
                       help="last season (start year), default: current season")
    fetch.set_defaults(run=cmd_fetch)

    ratings = commands.add_parser("ratings", help="show club ratings")
    ratings.add_argument("--as-of", type=parse_date, default=None,
                         help="use only matches before this date (YYYY-MM-DD)")
    ratings.set_defaults(run=cmd_ratings)

    match = commands.add_parser("match", help="predict one match")
    match.add_argument("home", help="home club (part of the name is enough)")
    match.add_argument("away", help="away club")
    match.add_argument("--as-of", type=parse_date, default=None,
                       help="use only matches before this date (YYYY-MM-DD)")
    match.set_defaults(run=cmd_match)
    return parser


def main():
    """Parse the command line and run the chosen sub-command."""
    # Windows consoles often default to a code page that cannot print "ş" or "ğ".
    sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
