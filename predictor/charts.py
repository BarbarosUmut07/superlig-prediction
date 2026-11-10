"""Matplotlib charts, saved as PNG files in output/."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# One dark colour scheme for every chart.
SURFACE = "#1a1a19"
TEXT = "#f0efec"
TEXT_SOFT = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
BLUE = "#3987e5"
ORANGE = "#d95926"
AQUA = "#199e70"
# Probabilities in heatmaps: dark = unlikely, bright = likely.
HEAT = LinearSegmentedColormap.from_list("heat", [SURFACE, "#104281", BLUE, "#cde2fb"])


def apply_theme():
    """Set the dark theme used by all charts."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": TEXT,
        "axes.labelcolor": TEXT_SOFT,
        "axes.edgecolor": GRID,
        "xtick.color": TEXT_SOFT,
        "ytick.color": TEXT_SOFT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    })


def save(fig, filename):
    """Save a figure to output/ at 150 dpi and return the path."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def percent(p):
    """Format a probability, never showing a misleading 0% or 100%."""
    if p < 0.01:
        return "<1%"
    if p > 0.99:
        return ">99%"
    return f"{p:.0%}"


def match_chart(model, home, away, filename=None):
    """Draw the outcome probabilities and a 6x6 score heatmap for one match."""
    apply_theme()
    lam, mu = model.expected_goals(home, away)
    probs = model.outcome_probs(home, away)
    scores = model.score_matrix(home, away)[:6, :6]

    fig, (bars, heat) = plt.subplots(1, 2, figsize=(11, 4.8),
                                     gridspec_kw={"width_ratios": [1, 1.15]})
    header(fig, f"{home} vs {away}",
           f"Expected goals: {home} {lam:.2f}, {away} {mu:.2f}")

    labels = [f"{home}\nwin", "Draw", f"{away}\nwin"]  # two lines: long names fit
    rects = bars.bar(labels, probs, color=[BLUE, MUTED, ORANGE], width=0.6)
    for rect, p in zip(rects, probs):
        bars.text(rect.get_x() + rect.get_width() / 2, p + 0.012, percent(p),
                  ha="center", fontsize=12, fontweight="bold")
    bars.set_ylim(0, max(probs) + 0.1)
    bars.set_yticks([])
    bars.spines["left"].set_visible(False)
    bars.tick_params(axis="x", length=0)

    heat.imshow(scores.T, cmap=HEAT, origin="lower", vmin=0)
    for x in range(6):
        for y in range(6):
            p = scores[x, y]
            colour = "#0b0b0b" if p > 0.75 * scores.max() else TEXT
            heat.text(x, y, f"{p:.1%}", ha="center", va="center", fontsize=8, color=colour)
    heat.set_xticks(range(6))
    heat.set_yticks(range(6))
    heat.set_xlabel(f"{home} goals")
    heat.set_ylabel(f"{away} goals")
    heat.set_title("Score probabilities", fontsize=10, color=TEXT_SOFT, loc="left")
    heat.tick_params(length=0)
    for spine in heat.spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(top=0.8, wspace=0.2)
    filename = filename or f"match_{home}_{away}.png".replace(" ", "_")
    return save(fig, filename)


def style_axis(ax, grid_axis=None):
    """Remove tick marks and, if wanted, draw faint grid lines behind the data."""
    ax.tick_params(length=0)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def header(fig, title, subtitle):
    """Write a left-aligned title and subtitle at the top of a figure."""
    height = fig.get_figheight()  # in inches, so the spacing is the same on every chart
    fig.text(0.02, 1 - 0.2 / height, title, fontsize=15, fontweight="bold", va="top")
    fig.text(0.02, 1 - 0.55 / height, subtitle, fontsize=10, color=TEXT_SOFT, va="top")


def forecast_chart(summary, title, subtitle, filename):
    """Draw the season forecast: points per club and the main probabilities.

    Left: points now, expected final points and the 5th-95th percentile range.
    Right: one column of bars each for champion, top 2, top 4 and relegation.
    """
    apply_theme()
    n = len(summary)
    y = np.arange(n)
    fig, axes = plt.subplots(1, 5, figsize=(13, 0.42 * n + 1.6), sharey=True,
                             gridspec_kw={"width_ratios": [3.4, 1, 1, 1, 1], "wspace": 0.08})
    header(fig, title, subtitle)

    points = axes[0]
    points.hlines(y, summary["points_p5"], summary["points_p95"], color=MUTED, linewidth=2,
                  label="90% range")
    points.scatter(summary["points_now"], y, s=38, facecolors=SURFACE, edgecolors=TEXT_SOFT,
                   linewidths=1.5, zorder=3, label="Points now")
    points.scatter(summary["exp_points"], y, s=55, color=BLUE, edgecolors=SURFACE,
                   linewidths=1.5, zorder=4, label="Expected final points")
    for row, value in zip(y, summary["exp_points"]):
        points.text(summary["points_p95"].max() + 3, row, f"{value:.0f}", va="center",
                    ha="left", fontsize=10, fontweight="bold")
    points.set_xlim(0, summary["points_p95"].max() + 9)
    points.set_yticks(y, summary["team"], fontsize=10, color=TEXT)
    points.set_ylim(n - 0.5, -0.5)  # best club at the top
    points.set_xlabel("Points")
    points.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3, frameon=False,
                  fontsize=9, handletextpad=0.3, columnspacing=1.2)
    points.spines["left"].set_visible(False)
    style_axis(points, grid_axis="x")

    columns = [("champion", "Champion", BLUE), ("top2", "Top 2", BLUE),
               ("top4", "Top 4", BLUE), ("relegation", "Relegation", ORANGE)]
    for ax, (column, label, colour) in zip(axes[1:], columns):
        ax.barh(y, summary[column], color=colour, height=0.62)
        for row, p in zip(y, summary[column]):
            # Bars end at 1.0; the numbers sit in their own strip to the right.
            ax.text(1.4, row, percent(p), va="center", ha="right", fontsize=9,
                    color=TEXT if p >= 0.01 else MUTED)
        ax.set_xlim(0, 1.45)
        ax.set_xticks([])
        ax.set_title(label, fontsize=10, color=TEXT_SOFT, pad=8, loc="left")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.axvline(0, color=GRID, linewidth=1)
        style_axis(ax)

    fig.subplots_adjust(top=1 - 1.3 / fig.get_figheight(), left=0.11, right=0.985,
                        bottom=0.7 / fig.get_figheight())
    return save(fig, filename)


def position_heatmap(position_probs, title, subtitle, filename):
    """Draw the chance of every final position (columns) for every club (rows)."""
    apply_theme()
    values = position_probs.to_numpy()
    n_clubs, n_places = values.shape
    fig, ax = plt.subplots(figsize=(0.52 * n_places + 2.4, 0.42 * n_clubs + 1.8))
    header(fig, title, subtitle)

    ax.imshow(values, cmap=HEAT, vmin=0, vmax=max(values.max(), 0.5), aspect="auto")
    for row in range(n_clubs):
        for column in range(n_places):
            p = values[row, column]
            if p >= 0.005:  # leave cells that round to 0% empty
                ax.text(column, row, f"{p * 100:.0f}", ha="center", va="center", fontsize=8,
                        color="#0b0b0b" if p > 0.6 else TEXT)
    ax.set_xticks(range(n_places), position_probs.columns)
    ax.set_yticks(range(n_clubs), position_probs.index, color=TEXT)
    ax.xaxis.tick_top()
    ax.set_xlabel("Final position (chance in %)")
    ax.xaxis.set_label_position("top")
    for spine in ax.spines.values():
        spine.set_visible(False)
    style_axis(ax)

    fig.subplots_adjust(top=1 - 1.45 / fig.get_figheight(), left=0.15, right=0.985, bottom=0.03)
    return save(fig, filename)


def ratings_chart(ratings, title, subtitle, filename="ratings.png"):
    """Draw attack and defence strength per club relative to the league average.

    Both panels are scaled so that a bar to the right is good: more goals
    scored than an average club, or fewer goals conceded.
    """
    apply_theme()
    n = len(ratings)
    y = np.arange(n)
    scored = (np.exp(ratings["attack"]) - 1) * 100       # % more goals scored
    conceded = (1 - np.exp(ratings["defence"])) * 100    # % fewer goals conceded
    limit = max(abs(scored).max(), abs(conceded).max()) * 1.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 0.42 * n + 1.6), sharey=True,
                             gridspec_kw={"wspace": 0.06})
    header(fig, title, subtitle)
    panels = [(scored, "Attack: goals scored vs average club", BLUE),
              (conceded, "Defence: goals conceded vs average club (right = fewer)", AQUA)]
    for ax, (values, label, colour) in zip(axes, panels):
        ax.barh(y, values, color=colour, height=0.62)
        for row, value in zip(y, values):
            offset = limit * 0.02 if value >= 0 else -limit * 0.02
            ax.text(value + offset, row, f"{value:+.0f}%", va="center", fontsize=9,
                    ha="left" if value >= 0 else "right")
        ax.axvline(0, color=TEXT_SOFT, linewidth=1)
        ax.set_xlim(-limit, limit)
        ax.set_title(label, fontsize=10, color=TEXT_SOFT, loc="left", pad=8)
        ax.set_xticks([])  # every bar carries its own number
        for spine in ax.spines.values():
            spine.set_visible(False)
        style_axis(ax)
    axes[0].set_yticks(y, ratings["team"], color=TEXT)
    axes[0].set_ylim(n - 0.5, -0.5)

    fig.subplots_adjust(top=1 - 1.25 / fig.get_figheight(), left=0.12, right=0.985,
                        bottom=0.2 / fig.get_figheight())
    return save(fig, filename)


def backtest_chart(scores, calibrations, title, subtitle, filename):
    """Draw the RPS of every forecaster and the calibration curves.

    `scores` is the table from evaluate.score_table; `calibrations` maps a
    forecaster name to its table from evaluate.calibration.
    """
    apply_theme()
    colours = {"Baseline": MUTED, "Dixon-Coles": BLUE, "Bookmakers": ORANGE}
    fig, (bars, curve) = plt.subplots(1, 2, figsize=(12, 5.2),
                                      gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.25})
    header(fig, title, subtitle)

    names = list(scores["forecaster"])
    rects = bars.bar(names, scores["rps"], width=0.6, color=[colours[name] for name in names])
    for rect, value in zip(rects, scores["rps"]):
        bars.text(rect.get_x() + rect.get_width() / 2, value + 0.003, f"{value:.4f}",
                  ha="center", fontsize=11, fontweight="bold")
    bars.set_ylim(0, scores["rps"].max() * 1.15)
    bars.set_title("Ranked probability score (lower is better)", fontsize=10,
                   color=TEXT_SOFT, loc="left", pad=8)
    bars.tick_params(axis="x", labelcolor=TEXT)
    style_axis(bars, grid_axis="y")

    curve.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--",
               label="Perfect calibration")
    for name, table in calibrations.items():
        table = table[table["count"] >= 10]  # a bin with a handful of cases is noise
        curve.plot(table["predicted"], table["observed"], color=colours[name], linewidth=2,
                   marker="o", markersize=7, markeredgecolor=SURFACE, label=name)
    curve.set_xlim(0, 1)
    curve.set_ylim(0, 1)
    curve.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    curve.set_aspect("equal")
    curve.set_xlabel("Predicted probability")
    curve.set_ylabel("How often it happened")
    curve.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    curve.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    curve.set_title("Calibration (bins with at least 10 cases)", fontsize=10, color=TEXT_SOFT, loc="left", pad=8)
    curve.legend(frameon=False, loc="upper left", fontsize=9)
    style_axis(curve, grid_axis="both")

    fig.subplots_adjust(top=0.76, left=0.06, right=0.97, bottom=0.11)
    return save(fig, filename)
