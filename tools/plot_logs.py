"""Compare columns from one or more training logs in one chart.

    python -m tools.plot_logs runs/selfplay/log.csv:win_rate runs/selfplay2/log.csv:win_rate
    python -m tools.plot_logs "vs self-play 1=runs/exploiter1/log.csv:eval_reference" \\
        "vs self-play 2=runs/exploiter2/log.csv:eval_reference" \\
        --percent --mark "0.5:even" --out docs/stage2_exploiters.png

Each series is [LABEL=]LOG:COLUMN. Without a label it is named after the run
folder and the column. Evaluation columns (eval_*) hold the latest evaluation
result, repeated until the next evaluation, so they are drawn as steps. Other
columns are measured afresh on every row and are smoothed with a moving average
(--smooth). Rows where a column has no value (nan) are left out.

Needs matplotlib (pip install matplotlib).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tools.plot_training import GRID, INK, INK_SOFT, SURFACE, read_log, smooth

COLORS = ["#2a78d6", "#eb6834", "#2f9e6e", "#8a5cc4", "#c9a227", "#d6457a"]


def parse_series(text: str) -> tuple[str, Path, str]:
    """'[LABEL=]LOG:COLUMN' -> (label, log path, column)."""
    label, sep, rest = text.partition("=")
    if not sep:
        label, rest = "", text
    path, sep, column = rest.rpartition(":")  # rpartition: Windows paths contain ':' too
    if not sep or not path or not column:
        raise argparse.ArgumentTypeError(f"expected [LABEL=]LOG:COLUMN, got {text!r}")
    path = Path(path)
    return label or f"{path.parent.name} {column}", path, column


def parse_mark(text: str) -> tuple[float, str]:
    value, _, label = text.partition(":")
    try:
        return float(value), label.strip()
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected VALUE[:LABEL], got {text!r}") from None


def curve(log: dict[str, np.ndarray], column: str, window: int) -> tuple[np.ndarray, np.ndarray, bool]:
    """x (millions of steps), y, and whether to draw it as steps."""
    if column not in log:
        raise SystemExit(f"no column {column!r}; the log has: {', '.join(log)}")
    keep = ~np.isnan(log[column])
    if not keep.any():
        raise SystemExit(f"column {column!r} has no values yet")
    x, y = log["step"][keep] / 1e6, log[column][keep]
    steps = column.startswith("eval_")
    return x, (y if steps else smooth(y, window)), steps


def spread(values: list[float], gap: float, top: float) -> list[float]:
    """Nudge label heights apart so that no two are closer than `gap`, staying below `top`."""
    order = np.argsort(values)
    placed = [float(v) for v in values]
    for below, above in zip(order, order[1:]):  # push upwards...
        placed[above] = max(placed[above], placed[below] + gap)
    placed[order[-1]] = min(placed[order[-1]], top)
    for above, below in zip(order[::-1], order[::-1][1:]):  # ...then back down from the top
        placed[below] = min(placed[below], placed[above] - gap)
    return placed


def plot(series: list[tuple[str, dict[str, np.ndarray], str]], out: Path, title: str = "",
         ylabel: str = "", percent: bool = False, marks: list[tuple[float, str]] = (),
         window: int = 5) -> None:
    import matplotlib
    matplotlib.use("Agg")  # draw to a file, no window
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                         "axes.labelcolor": INK_SOFT, "xtick.color": INK_SOFT,
                         "ytick.color": INK_SOFT})
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    fig.subplots_adjust(left=0.1, right=0.76, top=0.86 if title else 0.94, bottom=0.13)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(length=0)

    curves = [(label, *curve(log, column, window)) for label, log, column in series]
    end = max(x[-1] for _, x, _, _ in curves)
    for value, label in marks:
        ax.axhline(value, color=INK_SOFT, lw=0.8, ls=(0, (2, 3)))
        if label:
            ax.text(end * 0.01, value, f" {label}", color=INK_SOFT, va="bottom", fontsize=9)
    for (label, x, y, steps), color in zip(curves, COLORS * 2):
        ax.plot(x, y, color=color, lw=2, drawstyle="steps-post" if steps else "default")
    ax.set_xlim(0, end)
    if percent:
        ax.set_ylim(0, 1.04)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])

    # Label each line at the right edge, as close as possible to where it ends.
    low, high = ax.get_ylim()
    heights = spread([y[-1] for _, _, y, _ in curves], gap=0.07 * (high - low), top=high)
    for (label, _, y, _), height, color in zip(curves, heights, COLORS * 2):
        ax.plot([end * 1.01, end * 1.04], [height] * 2, color=color, lw=2, clip_on=False)
        ax.text(end * 1.05, height, label, color=INK, va="center")

    ax.set_xlabel("Training steps (millions)")
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        fig.suptitle(title, x=0.1, ha="left", y=0.97, fontsize=12, color=INK, fontweight="bold")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("series", type=parse_series, nargs="+", help="[LABEL=]LOG:COLUMN")
    parser.add_argument("--out", type=Path, default=Path("runs/compare.png"))
    parser.add_argument("--title", default="")
    parser.add_argument("--ylabel", default="")
    parser.add_argument("--percent", action="store_true", help="values are fractions from 0 to 1")
    parser.add_argument("--mark", type=parse_mark, action="append", default=[],
                        help='a dashed horizontal line, e.g. "0.5:even"')
    parser.add_argument("--smooth", type=int, default=5, help="moving-average window, in logged rows")
    args = parser.parse_args()
    logs = {path: read_log(path) for _, path, _ in args.series}
    series = [(label, logs[path], column) for label, path, column in args.series]
    plot(series, args.out, args.title, args.ylabel, args.percent, args.mark, args.smooth)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
