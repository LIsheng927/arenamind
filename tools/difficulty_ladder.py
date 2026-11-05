"""Build a difficulty ladder: one policy at several chosen strengths.

    python -m tools.difficulty_ladder runs/selfplay3/model.pt
    python -m tools.difficulty_ladder runs/selfplay3/model.pt --targets 1300 1100 900 700 --knob every

The top level is the policy at full strength. For each target rating, one knob
(every by default; see rl/difficulty.py) is turned until the policy's rating
against the fixed panel of rl/rate.py is within --tolerance of the target. The
search assumes the rating falls as the knob goes up. It takes the two nearest
settings tried on either side of the target and tries the level where a
straight line between their ratings crosses the target. Every try is kept, so
later targets start from what earlier ones found.

The levels are then checked against each other. They play a league among
themselves, and each pair's actual score is compared with the score their
panel ratings predict. If the ratings only held against the panel, it would
show here.

Writes to --out:

    ladder.json   the levels: setting, bot spec, rating and style
    tries.csv     every setting tried while searching
    checks.csv    every pair of levels: predicted and actual score
    ladder.png    rating against the knob with the levels marked, and actual
                  against predicted scores between levels

Needs PyTorch, and matplotlib for the chart.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path

import numpy as np

from rl.difficulty import Difficulty
from rl.league import run_league
from rl.rate import DEFAULT_PANEL, DEFAULT_PANEL_FILE, STYLE, panel_ratings, rate_player, short, write_csv

# Per knob: its full-strength level, the furthest level searched, and the step levels are rounded to.
SEARCH = {"every": (1.0, 5.0, 0.05), "temp": (1.0, 6.0, 0.05), "aim": (0.0, 60.0, 0.5)}
TARGETS = (1300, 1100, 900, 700)


def expected_score(a: float, b: float) -> float:
    """Share of the points a player rated `a` should take from one rated `b`."""
    return 1 / (1 + 10 ** ((b - a) / 400))


def find_setting(target: float, rate: Callable[[float], float], tried: dict[float, float],
                 step: float, tolerance: float = 25.0, max_tries: int = 6) -> float:
    """The knob level whose rating comes closest to `target`.

    `tried` maps the levels rated so far to their ratings and gains every new
    try; `rate(level)` rates a new level. The rating must fall as the level rises.
    """
    def closest() -> float:
        return min(tried, key=lambda x: abs(tried[x] - target))

    if not min(tried.values()) < target < max(tried.values()):
        raise ValueError(f"target {target:g} is outside the ratings tried so far")
    for _ in range(max_tries):
        if abs(tried[closest()] - target) <= tolerance:
            break
        levels = sorted(tried)
        # The first neighbouring pair, from the strong end, whose ratings straddle the target.
        a, b = next((a, b) for a, b in pairwise(levels) if tried[a] > target > tried[b])
        guess = a + (tried[a] - target) / (tried[a] - tried[b]) * (b - a)
        guess = min(max(guess, a + 0.15 * (b - a)), b - 0.15 * (b - a))  # keep off the ends
        guess = round(round(guess / step) * step, 6)
        guess = min(max(guess, round(a + step, 6)), round(b - step, 6))
        if not a < guess < b:  # the two are a single step apart
            break
        tried[guess] = rate(guess)
    return closest()


def pair_checks(names: list[str], ratings: list[float], score: np.ndarray,
                count: np.ndarray) -> list[dict]:
    """Each pair of levels (stronger first): predicted against actual share of the points."""
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            actual = score[i, j] / count[i, j]
            rows.append({"stronger": names[i], "weaker": names[j],
                         "rating_gap": round(ratings[i] - ratings[j], 1),
                         "predicted": round(expected_score(ratings[i], ratings[j]), 4),
                         "actual": round(actual, 4),
                         "noise": round(math.sqrt(actual * (1 - actual) / count[i, j]), 4),
                         "neighbours": j == i + 1})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint")
    parser.add_argument("--knob", choices=list(SEARCH), default="every")
    parser.add_argument("--targets", type=float, nargs="+", default=list(TARGETS),
                        help="ratings for the levels below full strength")
    parser.add_argument("--stop", type=float, help="the furthest knob level to search (default per knob)")
    parser.add_argument("--tolerance", type=float, default=25.0, help="how close to a target is enough")
    parser.add_argument("--max-tries", type=int, default=6, help="new settings tried per target at most")
    parser.add_argument("--games", type=int, default=100, help="games per panel member and side")
    parser.add_argument("--check-games", type=int, default=200,
                        help="games per pair of levels and side in the final check")
    parser.add_argument("--panel", nargs="+", default=list(DEFAULT_PANEL))
    parser.add_argument("--panel-games", type=int, default=200)
    parser.add_argument("--panel-file", default=DEFAULT_PANEL_FILE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/ladder"))
    args = parser.parse_args()

    panel = list(dict.fromkeys(args.panel))
    ratings = panel_ratings(panel, args.panel_games, args.seed, args.panel_file)
    print("panel: " + ", ".join(f"{short(p)} {r:.0f}" for p, r in ratings.items()))
    start, stop, step = SEARCH[args.knob]
    stop = args.stop or stop

    def setting(level: float) -> str:
        return Difficulty.parse(f"{args.knob}={level:g}").label()

    def spec(level: float) -> str:
        label = setting(level)
        return f"ppo:{args.checkpoint}" + ("" if label == "full" else f"@{label}")

    tried: dict[float, float] = {}
    details: dict[float, dict] = {}

    def rate(level: float) -> float:
        rating, scores, style = rate_player(spec(level), ratings, args.games, args.seed + 7)
        s = style.summary()
        details[level] = {"knob": args.knob, "level": level, "setting": setting(level),
                          "spec": spec(level), "elo": round(rating, 1),
                          **{key: round(s[key], 4) for key in STYLE},
                          **{f"vs {short(o)}": round(v, 4) for o, v in scores.items()}}
        print(f"  try {len(details):>2}: {setting(level):<12} Elo {rating:5.0f}")
        return rating

    print(f"the ends of the search ({args.knob} from {start:g} to {stop:g}):")
    for level in (start, stop):
        tried[level] = rate(level)
    targets = sorted(set(args.targets), reverse=True)
    outside = [t for t in targets if not tried[stop] < t < tried[start]]
    if outside:
        raise SystemExit(f"targets {', '.join(f'{t:g}' for t in outside)} are outside what "
                         f"{args.knob} reaches here ({tried[stop]:.0f} to {tried[start]:.0f})")

    chosen = [(None, start)]
    for target in targets:
        print(f"target {target:g}:")
        level = find_setting(target, rate, tried, step, args.tolerance, args.max_tries)
        chosen.append((target, level))
        print(f"  -> {setting(level)}, Elo {tried[level]:.0f}")

    # Strongest first; L1 is the weakest and the top level is full strength.
    names = [f"L{len(chosen) - k}" for k in range(len(chosen))]
    levels = [{"target": target, **details[level]} for target, level in chosen]

    print(f"check: the levels play each other ({args.check_games} games per pair and side)")
    score, count = run_league([lv["spec"] for lv in levels], args.check_games, args.seed + 11)
    checks = pair_checks(names, [lv["elo"] for lv in levels], score, count)

    full = levels[0]
    print(f"\n{'level':<6}{'setting':<13}{'target':>7}{'Elo':>7}  accuracy  reaction  turn rev/s"
          f"  blind shots")
    for name, lv in zip(names, levels):
        target = "-" if lv["target"] is None else f"{lv['target']:g}"
        ratio = lv["turn_reversals_per_s"] / full["turn_reversals_per_s"]
        extra = 100 * (lv["blind_shot_share"] - full["blind_shot_share"])
        print(f"{name:<6}{lv['setting']:<13}{target:>7}{lv['elo']:7.0f}  {lv['accuracy']:8.1%}"
              f"  {lv['reaction_s']:7.2f}s  {lv['turn_reversals_per_s']:5.2f} ({ratio:.2f}x)"
              f"  {lv['blind_shot_share']:5.1%} ({extra:+.1f})")
    print(f"\n{'pair':<9}{'gap':>6}  predicted  actual")
    for c in checks:
        print(f"{c['stronger'] + '-' + c['weaker']:<9}{c['rating_gap']:6.0f}  {c['predicted']:9.1%}"
              f"  {c['actual']:6.1%} +- {2 * c['noise']:.1%}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ladder.json").write_text(json.dumps({
        "checkpoint": args.checkpoint, "knob": args.knob, "games": args.games,
        "panel": {short(p): round(r, 1) for p, r in ratings.items()},
        "levels": [{"name": n, **lv} for n, lv in zip(names, levels)]}, indent=2))
    write_csv([details[x] for x in sorted(details)], args.out / "tries.csv")
    write_csv(checks, args.out / "checks.csv")
    plot(details, levels, names, checks, args.knob, args.out / "ladder.png")
    print(f"\nwrote ladder.json, tries.csv, checks.csv and ladder.png to {args.out}")


def plot(details: dict[float, dict], levels: list[dict], names: list[str], checks: list[dict],
         knob: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tools.difficulty_sweep import UNITS, style_axes
    from tools.plot_logs import COLORS
    from tools.plot_training import INK, INK_SOFT, SURFACE

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
                         "axes.labelcolor": INK_SOFT, "xtick.color": INK_SOFT,
                         "ytick.color": INK_SOFT})
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.4), facecolor=SURFACE,
                                      gridspec_kw={"width_ratios": [1.35, 1]})
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.2, wspace=0.22)

    # Left: every rating tried while searching, and the levels picked from them.
    style_axes(left)
    xs = sorted(details)
    left.plot(xs, [details[x]["elo"] for x in xs], color=INK_SOFT, lw=1, marker="o", ms=3)
    for name, lv in zip(names, levels):
        if lv["target"] is not None:
            left.axhline(lv["target"], color=INK_SOFT, lw=0.6, ls=(0, (2, 3)))
        left.plot(lv["level"], lv["elo"], "o", ms=8, color=COLORS[0], zorder=3)
        left.annotate(f"{name}  {lv['setting']}\n{lv['elo']:.0f}", (lv["level"], lv["elo"]),
                      xytext=(8, 4), textcoords="offset points", fontsize=8, color=INK)
    top = max(lv["elo"] for lv in levels)
    left.set_ylim(top=top + 0.15 * (top - min(d["elo"] for d in details.values())))
    left.set_xlabel(UNITS.get(knob, knob))
    left.set_ylabel("Elo against the fixed panel")
    left.set_title("Search: rating at each setting tried", loc="left", fontsize=10, color=INK)

    # Right: do the panel ratings predict how the levels do against each other?
    # Pairs of neighbouring levels first, then pairs two levels apart, and so on.
    style_axes(right)
    apart = {id(c): int(c["stronger"][1:]) - int(c["weaker"][1:]) for c in checks}
    order = sorted(checks, key=lambda c: (apart[id(c)], -int(c["stronger"][1:])))
    for x, c in enumerate(order):
        color = COLORS[1] if c["neighbours"] else INK_SOFT
        right.errorbar(x, c["actual"], yerr=2 * c["noise"], fmt="o", ms=5, color=color,
                       elinewidth=1, capsize=0, zorder=3)
        right.plot(x, c["predicted"], "D", ms=6, mfc="none", mec=INK, mew=1, zorder=4)
    right.plot([], [], "D", ms=6, mfc="none", mec=INK, mew=1, label="predicted by the panel ratings")
    right.errorbar([], [], yerr=[], fmt="o", ms=5, color=INK_SOFT,
                   label="actual (bar: \u00b12 standard errors)")
    right.set_xticks(range(len(order)))
    right.set_xticklabels([f"{c['stronger']}-{c['weaker']}" for c in order], rotation=45, ha="right",
                          fontsize=8)
    low = min(min(c["predicted"], c["actual"] - 2 * c["noise"]) for c in checks)
    right.set_ylim(max(0.0, low - 0.03), 1.01)
    right.set_ylabel("share of points for the stronger level")
    right.legend(frameon=False, loc="lower right", fontsize=8)
    right.set_title("Check: levels playing each other", loc="left", fontsize=10, color=INK)

    fig.suptitle("Difficulty ladder", x=0.07, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
