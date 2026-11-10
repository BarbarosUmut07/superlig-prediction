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
    fig.suptitle(f"{home} vs {away}", fontsize=15, fontweight="bold", x=0.02, ha="left")
    fig.text(0.02, 0.89, f"Expected goals: {home} {lam:.2f}, {away} {mu:.2f}",
             color=TEXT_SOFT, fontsize=10)

    labels = [f"{home} win", "Draw", f"{away} win"]
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
