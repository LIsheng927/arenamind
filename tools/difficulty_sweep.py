"""Measure how each difficulty knob changes a policy's strength and style.

    python -m tools.difficulty_sweep runs/selfplay3/model.pt
    python -m tools.difficulty_sweep runs/selfplay3/model.pt --knobs delay aim --games 50
    python -m tools.difficulty_sweep runs/selfplay3/model.pt --replot   # redraw from results.csv

Each knob is tried on its own at several levels (LEVELS below), starting from
full strength. Every setting is rated against the fixed panel of rl/rate.py
and its style is measured. Writes to --out:

    results.csv    one row per setting: rating, style, score against each panel member
    strength.png   rating against each knob's level, with the panel's ratings
    style.png      turn reversals per second and blind-shot share against rating

Needs PyTorch, and matplotlib for the charts.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from rl.difficulty import KNOBS, Difficulty
from rl.rate import (DEFAULT_PANEL, DEFAULT_PANEL_FILE, STYLE, panel_ratings, rate_player, short,
                     write_csv)
from tools.plot_logs import COLORS, spread
from tools.plot_training import GRID, INK, INK_SOFT, SURFACE

LEVELS = {"temp": (1.5, 2, 3, 5), "delay": (1, 2, 3, 5, 8), "aim": (2, 4, 8, 12, 20),
          "every": (2, 3, 5, 8)}
FULL = {"temp": 1, "delay": 0, "aim": 0, "every": 1}  # each knob's full-strength level
UNITS = {"temp": "temperature", "delay": "reaction delay (steps of 0.1 s)",
         "aim": "aim error (degrees)", "every": "decides every n steps"}


def settings(checkpoint: str, knobs: list[str]) -> list[tuple[str, float, str]]:
    """(knob, level, bot spec) for full strength and every level of the chosen knobs."""
    rows = [("full", 0, f"ppo:{checkpoint}")]
    for knob in knobs:
        for level in LEVELS[knob]:
            label = Difficulty.parse(f"{knob}={level}").label()
            rows.append((knob, level, f"ppo:{checkpoint}@{label}"))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint")
    parser.add_argument("--knobs", nargs="+", choices=list(KNOBS), default=list(KNOBS))
    parser.add_argument("--games", type=int, default=100, help="games per panel member and side")
    parser.add_argument("--panel", nargs="+", default=list(DEFAULT_PANEL))
    parser.add_argument("--panel-games", type=int, default=200)
    parser.add_argument("--panel-file", default=DEFAULT_PANEL_FILE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/difficulty"))
    parser.add_argument("--replot", action="store_true",
                        help="only redraw the charts from an earlier run's results.csv")
    args = parser.parse_args()

    panel = list(dict.fromkeys(args.panel))
    ratings = panel_ratings(panel, args.panel_games, args.seed, args.panel_file)
    print("panel: " + ", ".join(f"{short(p)} {r:.0f}" for p, r in ratings.items()))
    if args.replot:
        rows = read_rows(args.out / "results.csv")
        knobs = [k for k in KNOBS if any(r["knob"] == k for r in rows)]
        plot(rows, ratings, knobs, args.out)
        print(f"wrote {args.out / 'strength.png'} and {args.out / 'style.png'}")
        return

    rows = []
    todo = settings(args.checkpoint, args.knobs)
    for k, (knob, level, spec) in enumerate(todo, 1):
        rating, scores, style = rate_player(spec, ratings, args.games, args.seed + 7)
        s = style.summary()
        rows.append({"knob": knob, "level": level, "spec": spec, "elo": round(rating, 1),
                     **{key: round(s[key], 4) for key in STYLE},
                     **{f"vs {short(o)}": round(v, 4) for o, v in scores.items()}})
        print(f"[{k}/{len(todo)}] {short(spec):<28} Elo {rating:5.0f} | accuracy {s['accuracy']:.0%}, "
              f"reaction {s['reaction_s']:.2f}s, turn reversals {s['turn_reversals_per_s']:.2f}/s, "
              f"blind shots {s['blind_shot_share']:.0%}")
    write_csv(rows, args.out / "results.csv")
    plot(rows, ratings, args.knobs, args.out)
    print(f"wrote {args.out / 'results.csv'}, {args.out / 'strength.png'} and {args.out / 'style.png'}")


def read_rows(path: Path) -> list[dict]:
    """The rows of a results.csv written by an earlier run, with numbers as numbers."""
    def value(text: str) -> float | str:
        try:
            return float(text)
        except ValueError:
            return text
    with open(path, newline="") as f:
        return [{k: value(v) for k, v in row.items()} for row in csv.DictReader(f)]


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(length=0)


def curve(rows: list[dict], knob: str, x: str, y: str) -> tuple[list[float], list[float]]:
    """A knob's points from full strength outwards; x and y are column names ('level' -> knob level)."""
    full = next(r for r in rows if r["knob"] == "full")
    own = [r for r in rows if r["knob"] == knob]
    points = [(FULL[knob] if x == "level" else full[x], full[y])]
    points += [(r["level"] if x == "level" else r[x], r[y]) for r in own]
    return [p[0] for p in points], [p[1] for p in points]


def plot(rows: list[dict], ratings: dict[str, float], knobs: list[str], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
                         "axes.labelcolor": INK_SOFT, "xtick.color": INK_SOFT,
                         "ytick.color": INK_SOFT})
    out.mkdir(parents=True, exist_ok=True)

    # Strength: one panel per knob.
    fig, axes = plt.subplots(1, len(knobs), figsize=(3.2 * len(knobs) + 1.2, 3.6),
                             facecolor=SURFACE, sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.9, top=0.82, bottom=0.17, wspace=0.12)
    for ax, knob, color in zip(axes[0], knobs, COLORS):
        style_axes(ax)
        for name, r in ratings.items():
            ax.axhline(r, color=INK_SOFT, lw=0.6, ls=(0, (2, 3)))
        xs, ys = curve(rows, knob, "level", "elo")
        ax.plot(xs, ys, color=color, lw=2, marker="o", ms=4)
        ax.set_xlabel(UNITS[knob])
    last = axes[0][-1]
    x_end = last.get_xlim()[1]
    low, high = last.get_ylim()
    names = list(ratings)
    heights = spread([ratings[n] for n in names], gap=0.045 * (high - low), top=high)
    for name, height in zip(names, heights):  # labels nudged apart where ratings are close
        last.text(x_end, height, f" {short(name)}", va="center", fontsize=8, color=INK_SOFT,
                  clip_on=False)
    axes[0][0].set_ylabel("Elo against the fixed panel")
    fig.suptitle("Strength at each difficulty setting (one knob at a time)", x=0.07, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    fig.savefig(out / "strength.png", dpi=110, facecolor=SURFACE)
    plt.close(fig)

    # Style: does a weaker setting look twitchier?
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), facecolor=SURFACE)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.8, bottom=0.15, wspace=0.25)
    for ax, column, title in zip(axes, ("turn_reversals_per_s", "blind_shot_share"),
                                 ("Turn reversals per second", "Share of shots fired blind")):
        style_axes(ax)
        for knob, color in zip(knobs, COLORS):
            xs, ys = curve(rows, knob, "elo", column)
            ax.plot(xs, ys, color=color, lw=2, marker="o", ms=4, label=knob)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("Elo (weaker to the left)")
    axes[0].legend(frameon=False)
    fig.suptitle("How each knob changes the way it plays", x=0.08, ha="left", fontsize=11,
                 fontweight="bold", color=INK)
    fig.savefig(out / "style.png", dpi=110, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
