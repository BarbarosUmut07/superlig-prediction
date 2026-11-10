"""Streamlit app for the Süper Lig prediction model.

Run with:  streamlit run app.py
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from predictor import data, evaluate, simulate, table
from predictor.charts import AQUA, BLUE, MUTED, ORANGE, percent
from predictor.model import DixonColes

APP_SIMS = 5_000      # simulations for the forecast shown in the app
HISTORY_SIMS = 2_000  # simulations per date for the title-race line chart
RELEGATION = 3

st.set_page_config(page_title="Süper Lig Predictor", layout="wide")


# --- cached calculations ---------------------------------------------------
# st.cache_data stores the result for each combination of arguments, so moving
# a slider back to an earlier value does not recompute anything.

@st.cache_data
def load_matches():
    """Load data/matches.csv."""
    return data.load_matches()


@st.cache_data
def fit_model(as_of):
    """Fit the model on all matches before `as_of` (a date)."""
    return DixonColes().fit(load_matches(), as_of)


@st.cache_data
def run_forecast(season, as_of, n_sims):
    """Simulate the rest of `season` as seen on `as_of`."""
    matches = load_matches()
    teams = data.season_teams(matches, season)
    known = matches[(matches["season"] == season) & (matches["date"] < pd.Timestamp(as_of))]
    return simulate.simulate_season(fit_model(as_of), known, teams, n_sims, RELEGATION,
                                    table.load_deductions(season))


@st.cache_data
def title_history(season, last_day):
    """Title chance of every club every two weeks from the start of the season."""
    # Start the day after the first match, so the model always has something to learn from.
    first_day = load_matches().query("season == @season")["date"].min().date() + timedelta(days=1)
    days = pd.date_range(first_day, last_day, freq="14D").date.tolist()
    if days[-1] != last_day:
        days.append(last_day)
    rows = {day: run_forecast(season, day, HISTORY_SIMS)[0].set_index("team")["champion"]
            for day in days}
    return pd.DataFrame(rows).T


@st.cache_data
def run_backtest(season):
    """Walk-forward backtest of one season."""
    return evaluate.walk_forward(load_matches(), season)


# --- small helpers -----------------------------------------------------------

def dark(fig, height=420):
    """Apply the shared dark look to a plotly figure."""
    fig.update_layout(template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=40, b=10),
                      colorway=[BLUE, ORANGE, AQUA, "#c98500", "#d55181", "#9085e9"])
    return fig


def heatmap(values, x, y, x_title, y_title, height):
    """Blue heatmap of probabilities with the percentage written in each cell."""
    text = [[f"{p:.0%}" if p >= 0.005 else "" for p in row] for row in np.round(values, 2)]
    fig = go.Figure(go.Heatmap(
        z=values, x=x, y=y, text=text, texttemplate="%{text}", showscale=False,
        textfont=dict(color="#f0efec"),
        colorscale=[[0, "#1a1a19"], [0.5, "#104281"], [1, BLUE]],
        hovertemplate="%{y}, %{x}: %{z:.1%}<extra></extra>"))
    fig.update_xaxes(title=x_title, side="top", type="category")
    fig.update_yaxes(title=y_title, autorange="reversed", type="category")
    return dark(fig, height)


# --- pages -------------------------------------------------------------------

def page_forecast(matches, season):
    """Projected table, position heatmap, 'as of' slider and title-race chart."""
    st.title(f"Süper Lig {data.season_label(season)} forecast")
    rows = matches[matches["season"] == season]
    first_day = rows["date"].min().date() + timedelta(days=1)
    last_day = rows["date"].max().date() + timedelta(days=1)
    as_of = st.slider("Forecast as of", first_day, last_day, last_day, format="DD MMM YYYY",
                      help="The model only sees matches played before this date.")

    summary, position_probs = run_forecast(season, as_of, APP_SIMS)
    played = int((rows["date"] < pd.Timestamp(as_of)).sum())
    total = len(summary) * (len(summary) - 1)
    st.caption(f"{played} matches played before {as_of:%d %B %Y}; the other {total - played} "
               f"simulated {APP_SIMS:,} times.")

    shown = pd.DataFrame({
        "Club": summary["team"],
        "Played": summary["played"],
        "Points": summary["points_now"],
        "Expected points": summary["exp_points"].round(1),
        "90% range": [f"{low:.0f}-{high:.0f}"
                      for low, high in zip(summary["points_p5"], summary["points_p95"])],
        "Champion": summary["champion"].map(percent),
        "Top 2": summary["top2"].map(percent),
        "Top 4": summary["top4"].map(percent),
        "Relegation": summary["relegation"].map(percent),
    })
    shown.index = range(1, len(shown) + 1)
    st.dataframe(shown, height=35 * len(shown) + 38, width="stretch")

    st.subheader("Chance of every final position")
    st.plotly_chart(heatmap(position_probs.to_numpy(), list(position_probs.columns),
                            list(position_probs.index), "Final position", "", 620))

    st.subheader("Title race over the season")
    history = title_history(season, as_of)
    favourites = list(summary.sort_values("champion", ascending=False)["team"][:3])
    clubs = st.multiselect("Clubs", list(summary["team"]), default=favourites)
    fig = go.Figure()
    for club in clubs:
        fig.add_scatter(x=history.index, y=history[club], name=club, mode="lines+markers")
    fig.update_yaxes(title="Chance of winning the title", tickformat=".0%", range=[0, 1])
    st.plotly_chart(dark(fig))
    st.caption(f"One forecast every two weeks, {HISTORY_SIMS:,} simulations each. "
               "Each point uses only matches played before its date.")


def page_match(matches, season):
    """Pick two clubs and see the prediction."""
    st.title("Match predictor")
    teams = data.season_teams(matches, season)
    left, right = st.columns(2)
    home = left.selectbox("Home club", teams, index=teams.index("Galatasaray")
                          if "Galatasaray" in teams else 0)
    away_options = [team for team in teams if team != home]
    away = right.selectbox("Away club", away_options)

    model = fit_model(matches["date"].max().date() + timedelta(days=1))
    lam, mu = model.expected_goals(home, away)
    probs = model.outcome_probs(home, away)

    columns = st.columns(5)
    columns[0].metric(f"{home} win", f"{probs[0]:.0%}")
    columns[1].metric("Draw", f"{probs[1]:.0%}")
    columns[2].metric(f"{away} win", f"{probs[2]:.0%}")
    columns[3].metric(f"{home} expected goals", f"{lam:.2f}")
    columns[4].metric(f"{away} expected goals", f"{mu:.2f}")

    left, right = st.columns(2)
    bars = go.Figure(go.Bar(x=[f"{home} win", "Draw", f"{away} win"], y=probs,
                            marker_color=[BLUE, MUTED, ORANGE],
                            text=[f"{p:.0%}" for p in probs], textposition="outside"))
    bars.update_yaxes(tickformat=".0%", range=[0, max(probs) + 0.12])
    left.plotly_chart(dark(bars))

    scores = model.score_matrix(home, away)[:6, :6]
    right.plotly_chart(heatmap(scores.T[::-1], list(range(6)), list(range(5, -1, -1)),
                               f"{home} goals", f"{away} goals", 420))


def page_ratings(matches, season):
    """Attack against defence for every club of the season."""
    st.title("Club ratings")
    model = fit_model(matches["date"].max().date() + timedelta(days=1))
    ratings = model.ratings()
    ratings = ratings[ratings["team"].isin(data.season_teams(matches, season))]
    ratings = ratings.reset_index(drop=True)
    ratings.index += 1

    fig = go.Figure(go.Scatter(
        x=ratings["attack"], y=ratings["defence"], text=ratings["team"],
        mode="markers+text", textposition="top center", marker=dict(size=11, color=BLUE),
        hovertemplate="%{text}<br>attack %{x:.2f}<br>defence %{y:.2f}<extra></extra>"))
    fig.add_vline(x=0, line_color=MUTED, line_width=1)
    fig.add_hline(y=0, line_color=MUTED, line_width=1)
    fig.update_xaxes(title="Attack (right = scores more)")
    fig.update_yaxes(title="Defence (up = concedes fewer)", autorange="reversed")
    st.plotly_chart(dark(fig, 560))
    st.caption("0 is an average club. The best clubs are in the top-right corner. "
               f"Home advantage: x{np.exp(model.home_adv):.2f} goals; rho {model.rho:.3f}.")
    st.dataframe(ratings.round(2).rename(columns=str.capitalize), width="stretch")


def page_accuracy(matches, season):
    """Backtest results and calibration."""
    st.title("Model accuracy")
    # A season is finished when all n * (n - 1) matches are in the data.
    finished = []
    for s in sorted(matches["season"].unique()):
        n = len(data.season_teams(matches, s))
        if s >= 2018 and (matches["season"] == s).sum() == n * (n - 1):
            finished.append(int(s))
    chosen = st.selectbox("Season to test", finished[::-1], format_func=data.season_label)
    predictions = run_backtest(chosen)
    scores = evaluate.score_table(predictions)

    st.markdown(
        f"Every one of the {len(predictions)} matches of {data.season_label(chosen)} was "
        "predicted by a model that had only seen **earlier** matches. The model is compared "
        "with a naive baseline (the league's usual home/draw/away frequencies) and with "
        "bookmaker odds. The main score is the **ranked probability score**: lower is better, "
        "and it rewards probabilities that are close to what happened.")
    left, right = st.columns(2)
    colours = {"Baseline": MUTED, "Dixon-Coles": BLUE, "Bookmakers": ORANGE}
    bars = go.Figure(go.Bar(x=scores["forecaster"], y=scores["rps"],
                            marker_color=[colours[name] for name in scores["forecaster"]],
                            text=scores["rps"].map("{:.4f}".format), textposition="outside"))
    bars.update_layout(title="Ranked probability score (lower is better)")
    bars.update_yaxes(range=[0, scores["rps"].max() * 1.2])
    left.plotly_chart(dark(bars))

    curve = go.Figure()
    curve.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect",
                      line=dict(color=MUTED, dash="dash", width=1))
    for key in ("model", "book"):
        bins = evaluate.calibration(predictions, key)
        bins = bins[bins["count"] >= 10]
        name = evaluate.FORECASTERS[key]
        curve.add_scatter(x=bins["predicted"], y=bins["observed"], name=name,
                          mode="lines+markers", line=dict(color=colours[name]))
    curve.update_layout(title="Calibration", legend=dict(x=0.02, y=0.98))
    curve.update_xaxes(title="Predicted probability", tickformat=".0%", range=[0, 1])
    curve.update_yaxes(title="How often it happened", tickformat=".0%", range=[0, 1])
    right.plotly_chart(dark(curve))

    shown = scores.rename(columns={"forecaster": "Forecaster", "rps": "RPS",
                                   "log_loss": "Log loss", "brier": "Brier",
                                   "accuracy": "Accuracy"}).set_index("Forecaster")
    st.dataframe(shown.style.format({"RPS": "{:.4f}", "Log loss": "{:.4f}",
                                     "Brier": "{:.4f}", "Accuracy": "{:.1%}"}))
    st.markdown(
        "**How to read this.** The model is clearly better than the baseline and a little "
        "worse than the bookmakers. That is expected: bookmakers know about injuries, "
        "transfers and line-ups, while the model only sees results. On the calibration "
        "chart, points on the dashed line mean that events given, say, a 30% chance really "
        "happened about 30% of the time.")


def page_about(matches, season):
    """How the model works, the data and the limitations."""
    st.title("About")
    st.markdown(f"""
**The model.** A Dixon-Coles (1997) model. Every club has an attack and a defence strength;
goals are Poisson distributed with

- home goals: `λ = exp(intercept + home_adv + attack[home] + defence[away])`
- away goals: `μ = exp(intercept + attack[away] + defence[home])`

and a correction `τ` for the low scores 0-0, 1-0, 0-1 and 1-1. Recent matches count more
(a match one year old has about half the weight), and a ridge penalty pulls clubs with few
matches towards the league average.

**The forecast.** Every remaining fixture is simulated thousands of times from the model's
score probabilities. Each simulated table is ranked with the Süper Lig tie-break rules
(head-to-head first), and the percentages are simply how often something happened.

**The data.** {len(matches):,} Süper Lig matches from {matches['date'].min():%B %Y} to
{matches['date'].max():%d %B %Y}, results and bookmaker odds from
[football-data.co.uk](https://www.football-data.co.uk/turkeym.php). The 29 matches of
2022/23 that were awarded 3-0 after two clubs withdrew count for the table but not for
the model.

**Limitations.** The model only knows results. It knows nothing about injuries, transfers,
coaching changes or line-ups, so a club's improvement only shows after the results do.
Point deductions are only applied if they are listed in `data/deductions.csv`. The ranges
shown reflect the luck of the remaining matches, not the uncertainty about how good each
club really is.
""")


PAGES = {
    "Season forecast": page_forecast,
    "Match predictor": page_match,
    "Club ratings": page_ratings,
    "Model accuracy": page_accuracy,
    "About": page_about,
}


def main():
    """Draw the sidebar and the chosen page."""
    matches = load_matches()
    seasons = sorted(matches["season"].unique(), reverse=True)
    st.sidebar.title("Süper Lig Predictor")
    page = st.sidebar.radio("Page", list(PAGES))
    season = int(st.sidebar.selectbox("Season", seasons, format_func=data.season_label))
    st.sidebar.caption(f"Data up to {matches['date'].max():%d %B %Y}")
    PAGES[page](matches, season)


main()
