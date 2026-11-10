"""Command line for the Süper Lig prediction model.

Usage:
    python predict.py fetch [--first 2015] [--last 2026]
    python predict.py ratings [--as-of YYYY-MM-DD]
    python predict.py match <home> <away> [--as-of YYYY-MM-DD]
    python predict.py table [--season 2026] [--as-of YYYY-MM-DD]
    python predict.py simulate [--season 2026] [--as-of YYYY-MM-DD] [--sims 10000] [--relegation 3]
"""

import argparse
import sys
from datetime import date

import numpy as np

import pandas as pd

from predictor import charts, data, simulate, table
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


def season_matches(matches, season, as_of=None):
    """Return one season's matches, optionally only those before `as_of`."""
    rows = matches[matches["season"] == season]
    if as_of is not None:
        rows = rows[rows["date"] < pd.Timestamp(as_of)]
    return rows


def print_table(standings):
    """Print a league table built by table.league_table."""
    print(f"{'':>3} {'club':<19}{'P':>3}{'W':>4}{'D':>4}{'L':>4}{'GF':>5}{'GA':>4}{'GD':>5}{'Pts':>5}")
    for place, row in standings.iterrows():
        print(f"{place:>3} {row.team:<19}{row.played:>3}{row.won:>4}{row.drawn:>4}{row.lost:>4}"
              f"{row.gf:>5}{row.ga:>4}{row.gd:>+5}{row.points:>5}")


def cmd_table(args):
    """Print the league table of a season, with tie-breaks and deductions applied."""
    matches = data.load_matches()
    season = args.season if args.season is not None else season_on(matches, args.as_of)
    teams = data.season_teams(matches, season)
    deductions = table.load_deductions(season)
    print(f"Süper Lig {data.season_label(season)}"
          + (f", matches before {args.as_of}" if args.as_of else ""))
    print_table(table.league_table(season_matches(matches, season, args.as_of), teams, deductions))
    for team, points in deductions.items():
        print(f"{team}: {points} points deducted")


def print_forecast(summary):
    """Print the simulation summary, one row per club."""
    print(f"{'':>3} {'club':<16}{'P':>3}{'Pts':>5}{'xPts':>7}{'5%-95%':>9}{'xPos':>6}"
          f"{'Champ':>7}{'Top 2':>7}{'Top 4':>7}{'Releg':>7}")
    for place, row in enumerate(summary.itertuples(), start=1):
        spread = f"{row.points_p5:.0f}-{row.points_p95:.0f}"
        print(f"{place:>3} {row.team:<16}{row.played:>3}{row.points_now:>5}{row.exp_points:>7.1f}"
              f"{spread:>9}{row.exp_position:>6.1f}{charts.percent(row.champion):>7}"
              f"{charts.percent(row.top2):>7}{charts.percent(row.top4):>7}"
              f"{charts.percent(row.relegation):>7}")


def print_comparison(summary, final_table):
    """Print the forecast next to the real final table and the average points error."""
    real = final_table.reset_index(names="position").set_index("team")
    print(f"\n{'club':<16}{'real pos':>9}{'real pts':>9}{'forecast':>9}{'error':>7}")
    errors = []
    for team in real.index:
        expected = summary.set_index("team").loc[team, "exp_points"]
        error = expected - real.loc[team, "points"]
        errors.append(abs(error))
        print(f"{team:<16}{real.loc[team, 'position']:>9}{real.loc[team, 'points']:>9}"
              f"{expected:>9.1f}{error:>+7.1f}")
    print(f"\nAverage points error per club: {np.mean(errors):.1f}")


def cmd_simulate(args):
    """Simulate the rest of a season and print each club's chances."""
    matches = data.load_matches()
    season = args.season if args.season is not None else season_on(matches, args.as_of)
    teams = data.season_teams(matches, season)
    if not teams:
        sys.exit(f"Error: no matches for season {season}")
    deductions = table.load_deductions(season)
    known = season_matches(matches, season, args.as_of)
    model = fit_model(matches, args.as_of)

    left = len(teams) * (len(teams) - 1) - len(known)
    print(f"Süper Lig {data.season_label(season)}: {len(known)} matches played, "
          f"{left} simulated {args.sims} times\n")
    summary, position_probs = simulate.simulate_season(
        model, known, teams, args.sims, args.relegation, deductions)
    print_forecast(summary)

    charts.OUTPUT_DIR.mkdir(exist_ok=True)
    summary.to_csv(charts.OUTPUT_DIR / f"forecast_{season}.csv", index=False)
    position_probs.to_csv(charts.OUTPUT_DIR / f"positions_{season}.csv", index_label="team")
    print(f"\nSaved forecast_{season}.csv and positions_{season}.csv to {charts.OUTPUT_DIR}")

    whole_season = season_matches(matches, season)
    is_finished = len(whole_season) == len(teams) * (len(teams) - 1)
    if args.as_of is not None and is_finished:
        print_comparison(summary, table.league_table(whole_season, teams, deductions))


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

    standings = commands.add_parser("table", help="show a league table")
    standings.add_argument("--season", type=int, default=None, help="season (start year)")
    standings.add_argument("--as-of", type=parse_date, default=None,
                           help="use only matches before this date (YYYY-MM-DD)")
    standings.set_defaults(run=cmd_table)

    sim = commands.add_parser("simulate", help="simulate the rest of a season")
    sim.add_argument("--season", type=int, default=None, help="season (start year)")
    sim.add_argument("--as-of", type=parse_date, default=None,
                     help="pretend today is this date (YYYY-MM-DD)")
    sim.add_argument("--sims", type=int, default=10_000, help="number of simulations")
    sim.add_argument("--relegation", type=int, default=3, help="number of relegated clubs")
    sim.set_defaults(run=cmd_simulate)
    return parser


def main():
    """Parse the command line and run the chosen sub-command."""
    # Windows consoles often default to a code page that cannot print "ş" or "ğ".
    sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
