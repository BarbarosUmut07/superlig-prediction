"""Walk-forward backtest, scoring rules and calibration.

Three forecasters are compared on exactly the same matches:

    base   the league's home/draw/away frequencies in earlier seasons,
    model  the Dixon-Coles model, refitted before every match day,
    book   bookmaker odds turned into probabilities.

Probabilities are always in the order (home win, draw, away win) and the
outcome of a match is coded 0 = home win, 1 = draw, 2 = away win.
"""

import numpy as np
import pandas as pd

from predictor.model import DixonColes

FORECASTERS = {"base": "Baseline", "model": "Dixon-Coles", "book": "Bookmakers"}
SUFFIXES = ["h", "d", "a"]


def outcomes(matches):
    """Return the outcome code of every match: 0 home win, 1 draw, 2 away win."""
    hg, ag = matches["hg"].to_numpy(), matches["ag"].to_numpy()
    return np.where(hg > ag, 0, np.where(hg == ag, 1, 2))


def one_hot(outcome):
    """Turn outcome codes into rows like [0, 1, 0] (1 at what happened)."""
    return np.eye(3)[outcome]


def rps(probs, outcome):
    """Ranked probability score, averaged over matches (lower is better).

    Compares cumulative probabilities with the cumulative outcome, so the
    order home > draw > away matters: predicting a home win when the match is
    drawn is punished less than when the away side wins.
    """
    difference = np.cumsum(probs, axis=1) - np.cumsum(one_hot(outcome), axis=1)
    # The third cumulative value is always 1 - 1 = 0, so only two terms count.
    return float(np.mean(np.sum(difference[:, :2] ** 2, axis=1) / 2))


def log_loss(probs, outcome):
    """Average of -log(probability given to what happened) (lower is better)."""
    given = probs[np.arange(len(outcome)), outcome]
    return float(np.mean(-np.log(np.maximum(given, 1e-15))))


def brier(probs, outcome):
    """Average squared distance between the probabilities and the outcome (lower is better)."""
    return float(np.mean(np.sum((probs - one_hot(outcome)) ** 2, axis=1)))


def accuracy(probs, outcome):
    """Share of matches where the most likely outcome happened (higher is better)."""
    return float(np.mean(np.argmax(probs, axis=1) == outcome))


def bookmaker_probs(matches):
    """Turn decimal odds into probabilities.

    1/odds of the three outcomes sum to more than 1; the excess is the
    bookmaker's margin. Dividing by the sum removes it.
    """
    inverse = 1 / matches[["odds_h", "odds_d", "odds_a"]].to_numpy()
    return inverse / inverse.sum(axis=1, keepdims=True)


def baseline_probs(matches, season):
    """Return the home/draw/away frequencies of all seasons before `season`."""
    earlier = matches[(matches["season"] < season) & ~matches["awarded"]]
    return np.bincount(outcomes(earlier), minlength=3) / len(earlier)


def walk_forward(matches, season, xi=0.0019, ridge=0.5):
    """Predict every match of `season` using only matches played before its day.

    For each match day the model is refitted (warm start from the previous
    day) on all matches strictly before that day. Awarded matches and matches
    without odds are skipped so all forecasters are scored on the same games.
    Returns one row per match with the probabilities of the three forecasters.
    """
    test = matches[(matches["season"] == season) & ~matches["awarded"]]
    test = test.dropna(subset=["odds_h", "odds_d", "odds_a"])
    base = baseline_probs(matches, season)
    model = DixonColes(xi, ridge)

    days = []
    for day, day_matches in test.groupby("date"):
        model.fit(matches, as_of=day)
        model_probs = [model.outcome_probs(home, away)
                       for home, away in zip(day_matches["home"], day_matches["away"])]
        rows = day_matches[["date", "home", "away", "hg", "ag"]].copy()
        rows["outcome"] = outcomes(day_matches)
        rows[["model_h", "model_d", "model_a"]] = model_probs
        rows[["book_h", "book_d", "book_a"]] = bookmaker_probs(day_matches)
        rows[["base_h", "base_d", "base_a"]] = base
        days.append(rows)
    return pd.concat(days, ignore_index=True)


def probs_of(predictions, forecaster):
    """Return the (matches x 3) probability array of one forecaster."""
    return predictions[[f"{forecaster}_{s}" for s in SUFFIXES]].to_numpy()


def score_table(predictions):
    """Return RPS, log loss, Brier score and accuracy for every forecaster."""
    outcome = predictions["outcome"].to_numpy()
    rows = []
    for key, name in FORECASTERS.items():
        probs = probs_of(predictions, key)
        rows.append({
            "forecaster": name,
            "rps": rps(probs, outcome),
            "log_loss": log_loss(probs, outcome),
            "brier": brier(probs, outcome),
            "accuracy": accuracy(probs, outcome),
        })
    return pd.DataFrame(rows)


def calibration(predictions, forecaster, n_bins=5):
    """Compare predicted probabilities with how often the outcome happened.

    Every match gives three (probability, happened?) pairs, one per outcome.
    They are sorted into bins of equal width; for a well calibrated forecaster
    the average prediction of a bin equals the observed frequency.
    """
    predicted = probs_of(predictions, forecaster).ravel()
    happened = one_hot(predictions["outcome"].to_numpy()).ravel()
    bins = np.minimum((predicted * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        inside = bins == b
        if inside.any():
            rows.append({
                "bin": f"{b / n_bins:.0%}-{(b + 1) / n_bins:.0%}",
                "count": int(inside.sum()),
                "predicted": float(predicted[inside].mean()),
                "observed": float(happened[inside].mean()),
            })
    return pd.DataFrame(rows)


def grid_search(matches, season, xis, ridges):
    """Backtest the model on `season` for every (xi, ridge) pair.

    Returns a table sorted by RPS, best first. Use an earlier season here and
    judge the winner on a later one that played no part in the choice.
    """
    rows = []
    for xi in xis:
        for ridge in ridges:
            predictions = walk_forward(matches, season, xi, ridge)
            score = rps(probs_of(predictions, "model"), predictions["outcome"].to_numpy())
            rows.append({"xi": xi, "ridge": ridge, "rps": score})
    return pd.DataFrame(rows).sort_values("rps").reset_index(drop=True)
