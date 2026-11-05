"""Rate players against a fixed panel of opponents whose ratings are known.

    python -m rl.rate ppo:runs/selfplay3/model.pt ppo:runs/selfplay3/model.pt@delay=3
    python -m rl.rate camper --panel random rule rusher --games 200

A league (rl/league.py) rates everyone at once, so adding a player moves the
others. Here the panel is rated once, by a league among its own members, and
saved to a file; each player is then rated against the panel with the panel's
ratings held fixed. Ratings of different players, or of one policy at
different difficulty settings, can then be compared directly. Delete the
panel file to rate the panel again, for example after retraining one of its
models.

Each player's games are also measured for how it plays:

    accuracy        hits / shots fired
    reaction        seconds from the opponent coming into view to the first
                    shot (counted only when it shoots before losing sight)
    move changes    times per second the movement direction changes
    turn reversals  times per second it starts turning the other way
    blind shots     share of shots fired while the opponent is out of sight
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from arena.env import ArenaEnv
from tools.bot_spec import HELP, make_bot

from .league import ANCHOR, SCALE, fit_elo, run_league

DEFAULT_PANEL = ("random", "camper", "rule", "rusher", "ppo:runs/selfplay/model.pt",
                 "ppo:runs/selfplay2/model.pt", "ppo:runs/selfplay3/model.pt")
DEFAULT_PANEL_FILE = "runs/panel.json"
STYLE = ("accuracy", "reaction_s", "move_changes_per_s", "turn_reversals_per_s", "blind_shot_share")


@dataclass
class Style:
    """Counts gathered over a player's games; summary() turns them into rates."""

    seconds: float = 0.0
    shots: int = 0
    hits: int = 0
    blind_shots: int = 0
    move_changes: int = 0
    turn_reversals: int = 0
    reactions: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        def ratio(a: float, b: float) -> float:
            return a / b if b else float("nan")
        return {
            "accuracy": ratio(self.hits, self.shots),
            "reaction_s": float(np.mean(self.reactions)) if self.reactions else float("nan"),
            "move_changes_per_s": ratio(self.move_changes, self.seconds),
            "turn_reversals_per_s": ratio(self.turn_reversals, self.seconds),
            "blind_shot_share": ratio(self.blind_shots, self.shots),
        }


def play_measured(player: str, opponent: str, games: int, seed: int, side: int,
                  style: Style) -> tuple[int, int, int]:
    """Wins, losses and draws for `player` playing as `side`; adds its play to `style`."""
    env = ArenaEnv(games, seed=seed)
    bots = [None, None]
    bots[side] = make_bot(player, env, side, seed + 1)
    bots[1 - side] = make_bot(opponent, env, 1 - side, seed + 2)
    cfg = env.cfg
    if cfg.shoot_cooldown_steps < 2:
        raise ValueError("shots are recognised by the cooldown they start; it must be 2 steps or more")
    turn_sign = np.sign(np.array(cfg.agent.turn_degrees)).astype(np.int64)

    finished = np.zeros(games, dtype=bool)
    outcome = np.zeros(games, dtype=np.int64)
    prev_move = np.full(games, -1)
    last_turn = np.zeros(games, dtype=np.int64)  # direction of the last turn, 0 = none yet
    waiting = np.full(games, -1)  # steps since the opponent came into view; -1 = not waiting
    prev_visible = np.zeros(games, dtype=bool)
    while not finished.all():
        playing = ~finished
        visible = env.visible.copy()  # what both players can see when they decide
        waiting[playing & visible & ~prev_visible] = 0
        waiting[~visible] = -1
        actions = np.stack([b.act(env) for b in bots], axis=1)
        move, turn = actions[:, side, 0], turn_sign[actions[:, side, 1]]
        _, done, info = env.step(actions)
        # A shot sets the cooldown, which is then counted down once in the same step.
        fired = playing & (env.shoot_cd[:, side] == cfg.shoot_cooldown_steps - 1)

        style.seconds += cfg.dt * playing.sum()
        style.shots += int(fired.sum())
        style.blind_shots += int((fired & ~visible).sum())
        style.hits += int(np.rint(info["damage_dealt"][playing, side] / cfg.weapon.damage).sum())
        style.move_changes += int((playing & (prev_move >= 0) & (move != prev_move)).sum())
        reversal = playing & (turn != 0) & (last_turn != 0) & (turn != last_turn)
        style.turn_reversals += int(reversal.sum())
        counting = playing & (waiting >= 0)
        waiting[counting] += 1
        shot_back = counting & fired
        style.reactions.extend((waiting[shot_back] * cfg.dt).tolist())
        waiting[shot_back] = -1

        prev_move = np.where(playing, move, prev_move)
        last_turn = np.where(playing & (turn != 0), turn, last_turn)
        prev_visible = visible
        new = done & playing
        outcome[new] = info["outcome"][new, side]
        finished |= done
        prev_move[done], last_turn[done], waiting[done], prev_visible[done] = -1, 0, -1, False
        for b in bots:
            b.on_done(env, done)
    return int((outcome > 0).sum()), int((outcome < 0).sum()), int((outcome == 0).sum())


def panel_ratings(panel: list[str], games: int, seed: int, path: str | Path) -> dict[str, float]:
    """The panel's ratings from a league among its members, saved to `path` and reused."""
    path = Path(path)
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get("players") == panel and saved.get("games") == games:
            return saved["ratings"]
    score, count = run_league(panel, games, seed)
    anchor = panel.index(ANCHOR) if ANCHOR in panel else 0
    ratings = {p: float(r) for p, r in zip(panel, fit_elo(score, count, anchor))}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"players": panel, "games": games, "seed": seed,
                                "ratings": ratings}, indent=2))
    return ratings


def fit_against(score: dict[str, float], count: dict[str, float], ratings: dict[str, float],
                iterations: int = 100) -> float:
    """The rating under which the expected total score against the panel equals the actual one.

    The panel's ratings stay fixed. As in fit_elo, one draw is added against each
    opponent so that a perfect record still gives a finite rating.
    """
    names = list(score)
    opp = np.array([ratings[n] for n in names])
    s = np.array([score[n] for n in names]) + 0.5
    c = np.array([count[n] for n in names]) + 1.0
    rating = float(opp.mean())
    for _ in range(iterations):
        expected = 1 / (1 + 10 ** ((opp - rating) / 400))
        rating += SCALE * (s - c * expected).sum() / (c * expected * (1 - expected)).sum()
    return rating


def rate_player(player: str, ratings: dict[str, float], games: int,
                seed: int) -> tuple[float, dict[str, float], Style]:
    """Rating, score against each panel member and style, from `games` games per side."""
    style, score, count = Style(), {}, {}
    for k, opponent in enumerate(ratings):
        points = 0.0
        for side in (0, 1):
            w, _, d = play_measured(player, opponent, games, seed + 100 * k + side, side, style)
            points += w + 0.5 * d
        score[opponent], count[opponent] = points, 2 * games
    rating = fit_against(score, count, ratings)
    return rating, {o: score[o] / count[o] for o in score}, style


def short(spec: str) -> str:
    """'ppo:runs/selfplay3/model.pt@delay=3' -> 'selfplay3@delay=3'; bots keep their name."""
    if ":" not in spec:
        return spec
    path, at, settings = spec.split(":", 1)[1].partition("@")
    return Path(path).parent.name + at + settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("players", nargs="+", help=HELP)
    parser.add_argument("--panel", nargs="+", default=list(DEFAULT_PANEL))
    parser.add_argument("--games", type=int, default=100, help="games per panel member and side")
    parser.add_argument("--panel-games", type=int, default=200,
                        help="games per pair and side in the league that rates the panel")
    parser.add_argument("--panel-file", default=DEFAULT_PANEL_FILE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, help="also write the results to this CSV file")
    args = parser.parse_args()

    panel = list(dict.fromkeys(args.panel))
    ratings = panel_ratings(panel, args.panel_games, args.seed, args.panel_file)
    print("panel: " + ", ".join(f"{short(p)} {r:.0f}" for p, r in ratings.items()))

    rows = []
    width = max(len(short(p)) for p in args.players)
    print(f"{'player':<{width}}    Elo  accuracy  reaction  move ch/s  turn rev/s  blind shots")
    for player in args.players:
        rating, scores, style = rate_player(player, ratings, args.games, args.seed + 7)
        s = style.summary()
        print(f"{short(player):<{width}}  {rating:5.0f}  {s['accuracy']:8.0%}  {s['reaction_s']:7.2f}s"
              f"  {s['move_changes_per_s']:9.2f}  {s['turn_reversals_per_s']:10.2f}  {s['blind_shot_share']:11.0%}")
        rows.append({"player": player, "elo": round(rating, 1), **{k: round(v, 4) for k, v in s.items()},
                     **{f"vs {short(o)}": round(v, 4) for o, v in scores.items()}})
    if args.out:
        write_csv(rows, args.out)
        print(f"wrote {args.out}")


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
