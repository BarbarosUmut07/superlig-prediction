"""Command line for the Süper Lig prediction model.

Usage:
    python predict.py fetch [--first 2015] [--last 2026]
"""

import argparse
import sys

from predictor import data


def cmd_fetch(args):
    """Download all seasons, save data/matches.csv and print a summary."""
    matches = data.fetch_all(args.first, args.last)
    data.print_summary(matches)
    print(f"\nSaved {len(matches)} matches to {data.MATCHES_CSV}")


def build_parser():
    """Create the argument parser with one sub-command per task."""
    parser = argparse.ArgumentParser(description="Süper Lig prediction model")
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch", help="download results and build data/matches.csv")
    fetch.add_argument("--first", type=int, default=2015, help="first season (start year)")
    fetch.add_argument("--last", type=int, default=None,
                       help="last season (start year), default: current season")
    fetch.set_defaults(run=cmd_fetch)
    return parser


def main():
    """Parse the command line and run the chosen sub-command."""
    # Windows consoles often default to a code page that cannot print "ş" or "ğ".
    sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
