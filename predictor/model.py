"""Dixon-Coles (1997) model for football scores.

Every club has an attack and a defence strength. The goals of the two sides
are Poisson distributed with means

    lambda = exp(intercept + home_adv + attack[home] + defence[away])   (home goals)
    mu     = exp(intercept + attack[away] + defence[home])              (away goals)

and the probability of a score (x, y) is

    tau(x, y) * Poisson(x; lambda) * Poisson(y; mu)

where tau corrects the four low scores 0-0, 0-1, 1-0 and 1-1, which plain
independent Poisson goals get slightly wrong.

A positive attack means "scores more than average"; a positive defence means
"concedes more than average", so a good defence is negative.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10

# Strength of a club the model has never seen (just promoted, no history).
# The 33 clubs promoted to the Süper Lig since 2016 scored about 11% fewer and
# conceded about 6% more goals than the league average in their first season.
UNKNOWN_ATTACK = -0.12
UNKNOWN_DEFENCE = 0.06


def tau(x, y, lam, mu, rho):
    """Dixon-Coles correction factor for the score x-y.

    0-0:  1 - lam * mu * rho
    0-1:  1 + lam * rho
    1-0:  1 + mu * rho
    1-1:  1 - rho
    anything else: 1 (no correction)

    A negative rho makes 0-0 and 1-1 more likely and 1-0 and 0-1 less likely.
    """
    result = np.ones_like(lam, dtype=float)
    result = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, result)
    result = np.where((x == 0) & (y == 1), 1 + lam * rho, result)
    result = np.where((x == 1) & (y == 0), 1 + mu * rho, result)
    result = np.where((x == 1) & (y == 1), 1 - rho, result)
    return result


def tau_gradient(x, y, lam, mu, rho):
    """Derivatives of tau with respect to log(lam), log(mu) and rho."""
    zero = np.zeros_like(lam, dtype=float)
    d_log_lam, d_log_mu, d_rho = zero, zero, zero
    is_00 = (x == 0) & (y == 0)
    is_01 = (x == 0) & (y == 1)
    is_10 = (x == 1) & (y == 0)
    is_11 = (x == 1) & (y == 1)
    d_log_lam = np.where(is_00, -lam * mu * rho, d_log_lam)
    d_log_lam = np.where(is_01, lam * rho, d_log_lam)
    d_log_mu = np.where(is_00, -lam * mu * rho, d_log_mu)
    d_log_mu = np.where(is_10, mu * rho, d_log_mu)
    d_rho = np.where(is_00, -lam * mu, d_rho)
    d_rho = np.where(is_01, lam, d_rho)
    d_rho = np.where(is_10, mu, d_rho)
    d_rho = np.where(is_11, -1.0, d_rho)
    return d_log_lam, d_log_mu, d_rho


def negative_log_likelihood(params, home, away, hg, ag, weights, n_teams, ridge):
    """Return the penalised, weighted negative log-likelihood and its gradient.

    `params` is one flat vector: attack (n), defence (n), intercept, home_adv, rho.
    `home` and `away` are club indices, `hg` and `ag` the goals of each match.
    """
    attack = params[:n_teams]
    defence = params[n_teams:2 * n_teams]
    intercept, home_adv, rho = params[2 * n_teams:]

    log_lam = intercept + home_adv + attack[home] + defence[away]
    log_mu = intercept + attack[away] + defence[home]
    lam, mu = np.exp(log_lam), np.exp(log_mu)
    # tau can only be <= 0 for absurd parameter values; keep the log defined.
    t = np.maximum(tau(hg, ag, lam, mu, rho), 1e-10)

    # log of tau * Poisson(hg; lam) * Poisson(ag; mu), without the constant
    # -log(hg!) - log(ag!), which does not depend on the parameters.
    log_lik = np.log(t) + hg * log_lam - lam + ag * log_mu - mu
    penalty = ridge * (np.sum(attack ** 2) + np.sum(defence ** 2))
    value = -np.sum(weights * log_lik) + penalty

    # Gradient. For one match, d log_lik / d log_lam = (hg - lam) + tau part.
    t_lam, t_mu, t_rho = tau_gradient(hg, ag, lam, mu, rho)
    g_lam = weights * (hg - lam + t_lam / t)
    g_mu = weights * (ag - mu + t_mu / t)
    # attack[i] appears in lam when i is at home and in mu when i is away;
    # defence[i] appears in mu when i is at home and in lam when i is away.
    g_attack = np.bincount(home, g_lam, n_teams) + np.bincount(away, g_mu, n_teams)
    g_defence = np.bincount(home, g_mu, n_teams) + np.bincount(away, g_lam, n_teams)
    gradient = np.concatenate([
        -g_attack + 2 * ridge * attack,
        -g_defence + 2 * ridge * defence,
        [-np.sum(g_lam + g_mu), -np.sum(g_lam), -np.sum(weights * t_rho / t)],
    ])
    return value, gradient


class DixonColes:
    """Dixon-Coles model with time decay and a ridge penalty.

    xi:    time decay per day; a match d days old gets weight exp(-xi * d).
    ridge: penalty on attack² and defence², pulls clubs towards average.
    """

    def __init__(self, xi=0.0019, ridge=0.5):
        self.xi = xi
        self.ridge = ridge
        self.attack = {}
        self.defence = {}
        self.intercept = 0.0
        self.home_adv = 0.0
        self.rho = 0.0
        self.as_of = None

    def fit(self, matches, as_of=None, warm_start=True):
        """Fit the model on matches played strictly before `as_of`.

        `as_of` defaults to the day after the latest match. Awarded (unplayed)
        matches are ignored. With `warm_start` the previous parameters are the
        starting point, which makes refitting after a few new matches fast.
        """
        if as_of is None:
            as_of = matches["date"].max() + pd.Timedelta(days=1)
        as_of = pd.Timestamp(as_of)
        train = matches[matches["date"] < as_of]
        if "awarded" in train.columns:
            train = train[~train["awarded"]]
        if train.empty:
            raise ValueError(f"no matches before {as_of:%Y-%m-%d}")

        teams = sorted(set(train["home"]) | set(train["away"]))
        index = {team: i for i, team in enumerate(teams)}
        home = train["home"].map(index).to_numpy()
        away = train["away"].map(index).to_numpy()
        hg = train["hg"].to_numpy()
        ag = train["ag"].to_numpy()
        days_ago = (as_of - train["date"]).dt.days.to_numpy()
        weights = np.exp(-self.xi * days_ago)

        n = len(teams)
        start = np.zeros(2 * n + 3)
        start[2 * n:] = [0.2, 0.25, 0.0]  # intercept, home_adv, rho
        if warm_start and self.attack:
            start[:n] = [self.attack.get(team, 0.0) for team in teams]
            start[n:2 * n] = [self.defence.get(team, 0.0) for team in teams]
            start[2 * n:] = [self.intercept, self.home_adv, self.rho]

        bounds = [(None, None)] * (2 * n + 2) + [(-0.3, 0.3)]
        result = minimize(
            negative_log_likelihood,
            start,
            args=(home, away, hg, ag, weights, n, self.ridge),
            jac=True,  # the function returns (value, gradient)
            method="L-BFGS-B",
            bounds=bounds,
        )
        self.attack = dict(zip(teams, result.x[:n]))
        self.defence = dict(zip(teams, result.x[n:2 * n]))
        self.intercept, self.home_adv, self.rho = result.x[2 * n:]
        self.as_of = as_of
        return self

    def expected_goals(self, home, away):
        """Return (lambda, mu): expected goals of the home and the away club."""
        home_attack = self.attack.get(home, UNKNOWN_ATTACK)
        home_defence = self.defence.get(home, UNKNOWN_DEFENCE)
        away_attack = self.attack.get(away, UNKNOWN_ATTACK)
        away_defence = self.defence.get(away, UNKNOWN_DEFENCE)
        lam = np.exp(self.intercept + self.home_adv + home_attack + away_defence)
        mu = np.exp(self.intercept + away_attack + home_defence)
        return float(lam), float(mu)

    def score_matrix(self, home, away):
        """Return an 11x11 array: element [x, y] is the probability of the score x-y."""
        lam, mu = self.expected_goals(home, away)
        goals = np.arange(MAX_GOALS + 1)
        matrix = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
        x, y = np.meshgrid(goals, goals, indexing="ij")
        matrix = matrix * np.maximum(tau(x, y, lam, mu, self.rho), 0.0)
        # Scores above 10 goals are cut off, so rescale to sum to exactly 1.
        return matrix / matrix.sum()

    def outcome_probs(self, home, away):
        """Return the probabilities (home win, draw, away win)."""
        matrix = self.score_matrix(home, away)
        home_win = np.tril(matrix, -1).sum()  # below the diagonal: x > y
        draw = np.trace(matrix)
        away_win = np.triu(matrix, 1).sum()
        return float(home_win), float(draw), float(away_win)

    def ratings(self):
        """Return attack, defence and overall rating per club, best club first.

        The overall rating is attack - defence: a club is good when it scores
        a lot (high attack) and concedes little (low defence).
        """
        table = pd.DataFrame({
            "team": list(self.attack),
            "attack": list(self.attack.values()),
            "defence": list(self.defence.values()),
        })
        table["rating"] = table["attack"] - table["defence"]
        return table.sort_values("rating", ascending=False).reset_index(drop=True)
