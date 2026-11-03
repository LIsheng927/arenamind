"""Plot a training run's progress from its log.csv.

    python -m tools.plot_training runs/first/log.csv --out docs/stage1_training.png
    python -m tools.plot_training runs/first/log.csv --phase "0:stop dying" --phase "1.3:learn to hit"

Three panels share the x axis (training steps): win rate, hits dealt and taken
per game, and game length. Each logged row covers only a few hundred games, so
the curves are smoothed with a short moving average. The x axis uses a
square-root scale, because most of the learning happens early on.

Needs matplotlib (pip install matplotlib).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from arena.config import DEFAULT_CONFIG

SURFACE, INK, INK_SOFT, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
NICE_TICKS = [0, 0.5, 1, 2, 3, 5, 8, 12, 16, 20, 30, 50, 80, 100]


def read_log(path: Path) -> dict[str, np.ndarray]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} has no rows yet")
    return {key: np.array([float(r[key]) for r in rows]) for key in rows[0]}


def smooth(x: np.ndarray, k: int) -> np.ndarray:
    """Moving average over the last k rows (the first rows average what exists)."""
    k = max(1, min(k, len(x)))
    padded = np.concatenate([np.full(k - 1, x[0]), x])
    return np.convolve(padded, np.ones(k) / k, mode="valid")


def parse_phase(text: str) -> tuple[float, str]:
    start, _, label = text.partition(":")
    try:
        return float(start), label.strip()
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected MILLIONS:LABEL, got {text!r}") from None


def plot(log: dict[str, np.ndarray], out: Path, title: str = "",
         phases: list[tuple[float, str]] = (), window: int = 5, cfg=DEFAULT_CONFIG) -> None:
    import matplotlib
    matplotlib.use("Agg")  # draw to a file, no window
    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtransforms

    steps = log["step"] / 1e6
    end = steps[-1]
    hits_to_kill = cfg.agent.max_hp / cfg.weapon.damage
    panels = [
        ("Win rate", [(smooth(log["win_rate"], window) * 100, BLUE, None)], 100, "%"),
        (f"Hits per game ({hits_to_kill:g} hits kill)",
         [(smooth(log["hits_per_game"], window), BLUE, "dealt"),
          (smooth(log["hit_by_per_game"], window), ORANGE, "taken")], hits_to_kill, ""),
        (f"Game length (seconds; {cfg.max_time:g} = time limit)",
         [(smooth(log["mean_game_steps"], window) * cfg.dt, BLUE, None)], cfg.max_time, ""),
    ]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                         "axes.labelcolor": INK_SOFT, "xtick.color": INK_SOFT,
                         "ytick.color": INK_SOFT})
    fig, axes = plt.subplots(3, 1, figsize=(8, 7.2), sharex=True, facecolor=SURFACE)
    top = 0.86 if phases else 0.9
    fig.subplots_adjust(left=0.09, right=0.84, top=top, bottom=0.08, hspace=0.42)
    ticks = [t for t in NICE_TICKS if t <= end]

    for ax, (name, lines, ymax, unit) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(length=0)
        ax.set_xscale("function", functions=(np.sqrt, np.square))
        ax.set_xlim(0, end)
        for start, _ in phases:
            if start > 0:
                ax.axvline(start, color=INK_SOFT, lw=0.8, ls=(0, (2, 3)))
        for values, color, label in lines:
            ax.plot(steps, values, color=color, lw=2)
            if label:  # label the line at its right end
                ax.plot([end * 1.01, end * 1.04], [values[-1]] * 2, color=color, lw=2, clip_on=False)
                ax.text(end * 1.05, values[-1], label, color=INK, va="center")
        ax.set_ylim(0, ymax * 1.08)
        if unit == "%":
            ax.set_yticks([0, 50, 100], ["0%", "50%", "100%"])
        ax.set_title(name, loc="left", fontsize=11, color=INK)
    axes[-1].set_xticks(ticks, [f"{t:g}" for t in ticks])
    axes[-1].set_xlabel("Training steps (millions, square-root scale)")

    # Phase names, centred over their stretch of the (square-root) x axis.
    band = mtransforms.blended_transform_factory(axes[0].transData, axes[0].transAxes)
    bounds = [start for start, _ in phases] + [end]
    for (start, label), stop in zip(phases, bounds[1:]):
        middle = ((np.sqrt(start) + np.sqrt(stop)) / 2) ** 2
        axes[0].text(middle, 1.2, label, transform=band, ha="center", va="bottom",
                     fontsize=9.5, color=INK_SOFT)
    if title:
        fig.suptitle(title, x=0.09, ha="left", y=0.985, fontsize=12, color=INK,
                     fontweight="bold")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log", type=Path, help="a run's log.csv")
    parser.add_argument("--out", type=Path, help="image to write (default: next to the log)")
    parser.add_argument("--title", default="")
    parser.add_argument("--phase", type=parse_phase, action="append", default=[],
                        help='mark a phase starting at this many million steps, e.g. "1.3:learn to hit"')
    parser.add_argument("--smooth", type=int, default=5, help="moving-average window, in logged rows")
    args = parser.parse_args()
    out = args.out or args.log.with_name("training.png")
    plot(read_log(args.log), out, args.title, sorted(args.phase), args.smooth)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
