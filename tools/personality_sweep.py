"""Measure how each personality setting changes the way a policy plays, and what it costs.

    python -m tools.personality_sweep runs/selfplay3/model.pt
    python -m tools.personality_sweep runs/selfplay3/model.pt --traits hold --games 50
    python -m tools.personality_sweep runs/selfplay3/model.pt --replot   # redraw from results.csv

Each setting of rl/personality.py is tried on its own at several levels
(LEVELS below), at full strength. Every setting is rated against the fixed
panel of rl/rate.py and its play is measured. Writes to --out:

    results.csv       one row per setting: rating, style, score against each panel member
    personality.png   rating and the measures of play against each trait's level

A trait passes if it moves the measure it is meant to move by the amount below
while costing at most MAX_COST rating points (both fixed before the first run):

    approach  distance to the opponent at least 15% shorter at some positive
              level and at least 15% longer at some negative level
    hold      share of blind shots at most half of full strength's
    dash      dashes per second at least doubled or at least halved
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rl.personality import TRAITS, Personality
from rl.rate import (DEFAULT_PANEL, DEFAULT_PANEL_FILE, STYLE, panel_ratings, rate_player, short,
                     write_csv)
from tools.difficulty_sweep import read_rows, style_axes
from tools.plot_logs import COLORS
from tools.plot_training import GRID, INK, INK_SOFT, SURFACE

LEVELS = {"approach": (-3, -1.5, 1.5, 3), "hold": (1, 2, 4, 8), "dash": (-3, -1.5, 1.5, 3)}
MAX_COST = 100.0
ROWS = {"elo": "Elo against the panel", "distance": "distance to opponent", "in_sight_share":
        "share of time in sight", "blind_shot_share": "share of shots fired blind",
        "dashes_per_s": "dashes per second", "hits_taken_per_game": "hits taken per game"}


def settings(checkpoint: str, traits: list[str]) -> list[tuple[str, float, str]]:
    """(trait, level, bot spec) for full strength and every level of the chosen traits."""
    rows = [("full", 0, f"ppo:{checkpoint}")]
    for trait in traits:
        for level in LEVELS[trait]:
            rows.append((trait, level, f"ppo:{checkpoint}@{Personality(**{trait: level}).label()}"))
    return rows


def verdicts(rows: list[dict]) -> dict[str, tuple[bool, str]]:
    """For each trait swept: did it pass, and the evidence (see the module docstring)."""
    full = next(r for r in rows if r["trait"] == "full")
    result = {}
    for trait in TRAITS:
        own = [r for r in rows if r["trait"] == trait]
        if not own:
            continue
        cheap = [r for r in own if full["elo"] - r["elo"] <= MAX_COST]

        def change(r: dict, key: str) -> float:
            return r[key] / full[key]

        if trait == "approach":
            closer = [r["level"] for r in cheap if r["level"] > 0 and change(r, "distance") <= 0.85]
            farther = [r["level"] for r in cheap if r["level"] < 0 and change(r, "distance") >= 1.15]
            ok = bool(closer and farther)
            why = f"15% closer at {closer or 'none'}, farther at {farther or 'none'}"
        elif trait == "hold":
            halved = [r["level"] for r in cheap if change(r, "blind_shot_share") <= 0.5]
            ok, why = bool(halved), f"blind shots halved at {halved or 'none'}"
        else:
            moved = [r["level"] for r in cheap if not 0.5 < change(r, "dashes_per_s") < 2]
            ok, why = bool(moved), f"dashes doubled or halved at {moved or 'none'}"
        result[trait] = (ok, why + f" (within {MAX_COST:.0f} points: {[r['level'] for r in cheap]})")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint")
    parser.add_argument("--traits", nargs="+", choices=list(TRAITS), default=list(TRAITS))
    parser.add_argument("--games", type=int, default=100, help="games per panel member and side")
    parser.add_argument("--panel", nargs="+", default=list(DEFAULT_PANEL))
    parser.add_argument("--panel-games", type=int, default=200)
    parser.add_argument("--panel-file", default=DEFAULT_PANEL_FILE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/personality"))
    parser.add_argument("--replot", action="store_true",
                        help="only redraw the chart from an earlier run's results.csv")
    args = parser.parse_args()

    panel = list(dict.fromkeys(args.panel))
    ratings = panel_ratings(panel, args.panel_games, args.seed, args.panel_file)
    print("panel: " + ", ".join(f"{short(p)} {r:.0f}" for p, r in ratings.items()))
    if args.replot:
        rows = read_rows(args.out / "results.csv")
    else:
        rows = []
        todo = settings(args.checkpoint, args.traits)
        for k, (trait, level, spec) in enumerate(todo, 1):
            rating, scores, style = rate_player(spec, ratings, args.games, args.seed + 7)
            s = style.summary()
            rows.append({"trait": trait, "level": level, "spec": spec, "elo": round(rating, 1),
                         **{key: round(s[key], 4) for key in STYLE},
                         **{f"vs {short(o)}": round(v, 4) for o, v in scores.items()}})
            print(f"[{k}/{len(todo)}] {short(spec):<24} Elo {rating:5.0f} | distance {s['distance']:5.2f}, "
                  f"in sight {s['in_sight_share']:.0%}, blind shots {s['blind_shot_share']:.0%}, "
                  f"dashes {s['dashes_per_s']:.2f}/s, hits taken {s['hits_taken_per_game']:.2f}/game")
        write_csv(rows, args.out / "results.csv")
    traits = [t for t in TRAITS if any(r["trait"] == t for r in rows)]
    plot(rows, traits, args.out / "personality.png")
    for trait, (ok, why) in verdicts(rows).items():
        print(f"{trait:<9} {'passes' if ok else 'fails '}  {why}")
    print(f"wrote {args.out / 'results.csv'} and {args.out / 'personality.png'}")


def plot(rows: list[dict], traits: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
                         "axes.labelcolor": INK_SOFT, "xtick.color": INK_SOFT,
                         "ytick.color": INK_SOFT})
    full = next(r for r in rows if r["trait"] == "full")
    size = (3.3 * len(traits) + 1.6, 1.9 * len(ROWS) + 0.8)
    fig, axes = plt.subplots(len(ROWS), len(traits), figsize=size, facecolor=SURFACE, sharey="row",
                             squeeze=False)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.93, bottom=0.06, hspace=0.35, wspace=0.12)
    for col, (trait, color) in enumerate(zip(traits, COLORS)):
        points = sorted([(0.0, full)] + [(r["level"], r) for r in rows if r["trait"] == trait],
                        key=lambda p: p[0])
        for row, key in enumerate(ROWS):
            ax = axes[row][col]
            style_axes(ax)
            ax.axhline(full[key], color=GRID, lw=1)
            ax.plot([p[0] for p in points], [p[1][key] for p in points], color=color, lw=2,
                    marker="o", ms=4)
            if row == 0:
                ax.set_title(trait, loc="left", fontsize=10, color=INK)
            if row == len(ROWS) - 1:
                ax.set_xlabel(f"{trait} (0 = the policy's own play)")
            if col == 0:
                ax.set_ylabel(ROWS[key], rotation=0, ha="right", va="center")
    fig.suptitle("Personality settings at full strength", x=0.02, ha="left", fontsize=11,
                 fontweight="bold", color=INK)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
