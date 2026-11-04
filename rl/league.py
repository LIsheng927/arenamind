"""Rate players by playing them against each other: a small Elo league.

    python -m rl.league rule camper rusher kiter random ppo:runs/first/model.pt
    python -m rl.league rule ppo:runs/first/model.pt ppo-greedy:runs/first/model.pt --games 400

Every pair of players meets for `games` games with each of them as blue, then
`games` more with the sides swapped. A player can be any bot name or a trained
policy (see tools/bot_spec.py).

Ratings follow the Elo model: A's expected score against B is

    E = 1 / (1 + 10 ** ((R_B - R_A) / 400))

(a win counts 1, a draw 0.5). Instead of updating ratings game by game, which
depends on the order of the games, they are fitted to all results at once: the
ratings are chosen so that every player's expected total score equals its
actual total score. One draw is added to every pair, so that a perfect record
does not push a rating to infinity. Only differences matter, so the rule bot
is pinned at 1000 when it takes part (otherwise the first player is).
"""

from __future__ import annotations

import argparse
import itertools
import math

import numpy as np

from arena.env import ArenaEnv
from tools.bot_spec import HELP, make_bot

SCALE = 400 / math.log(10)  # Elo points per unit of the logistic curve
ANCHOR, ANCHOR_RATING = "rule", 1000.0


def play(blue: str, orange: str, games: int, seed: int) -> tuple[int, int, int]:
    """Wins, losses and draws for `blue` over `games` games (all run at once)."""
    env = ArenaEnv(games, seed=seed)
    bots = [make_bot(blue, env, 0, seed + 1), make_bot(orange, env, 1, seed + 2)]
    finished = np.zeros(games, dtype=bool)
    outcome = np.zeros(games, dtype=np.int64)
    while not finished.all():
        actions = np.stack([b.act(env) for b in bots], axis=1)
        _, done, info = env.step(actions)
        new = done & ~finished
        outcome[new] = info["outcome"][new, 0]
        finished |= done
        for b in bots:
            b.on_done(env, done)
    return int((outcome > 0).sum()), int((outcome < 0).sum()), int((outcome == 0).sum())


def run_league(players: list[str], games: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Score and game-count matrices: score[i, j] = points i took from j."""
    n = len(players)
    score, count = np.zeros((n, n)), np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        for k, (a, b) in enumerate([(i, j), (j, i)]):  # each player once as blue
            w, l, d = play(players[a], players[b], games, seed + 1000 * a + 10 * b + k)
            score[a, b] += w + 0.5 * d
            score[b, a] += l + 0.5 * d
            count[a, b] += games
            count[b, a] += games
    return score, count


def fit_elo(score: np.ndarray, count: np.ndarray, anchor: int = 0,
            anchor_rating: float = ANCHOR_RATING, iterations: int = 500) -> np.ndarray:
    """Ratings under which every player's expected total score equals its actual one."""
    n = len(score)
    off_diagonal = ~np.eye(n, dtype=bool)
    score = score + 0.5 * off_diagonal  # one extra draw per pair
    count = count + 1.0 * off_diagonal
    rating = np.zeros(n)
    for _ in range(iterations):
        expected = 1 / (1 + 10 ** ((rating[None, :] - rating[:, None]) / 400))
        surprise = (score - count * expected).sum(axis=1)  # actual minus expected, per player
        curvature = (count * expected * (1 - expected)).sum(axis=1)
        rating += SCALE * surprise / curvature  # Newton step, one player at a time
    return rating - rating[anchor] + anchor_rating


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("players", nargs="+", help=HELP)
    parser.add_argument("--games", type=int, default=200, help="games per pair and side")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    players = list(dict.fromkeys(args.players))  # drop repeats, keep order
    if len(players) < 2:
        raise SystemExit("a league needs at least two players")

    score, count = run_league(players, args.games, args.seed)
    anchor = players.index(ANCHOR) if ANCHOR in players else 0
    rating = fit_elo(score, count, anchor)

    order = np.argsort(-rating)
    width = max(len(p) for p in players)
    print(f"{'player':<{width}}  {'Elo':>6}  score")
    for i in order:
        print(f"{players[i]:<{width}}  {rating[i]:6.0f}  {score[i].sum() / count[i].sum():6.1%}")
    print(f"\nscore of row against column ({2 * args.games} games per pair)")
    tags = [f"#{k + 1}" for k in range(len(players))]
    print(" " * (width + 4) + "".join(f"{t:>7}" for t in tags))
    for k, i in enumerate(order):
        cells = "".join("      -" if i == j else f"{score[i, j] / count[i, j]:7.0%}" for j in order)
        print(f"#{k + 1:<3}{players[i]:<{width}}{cells}")


if __name__ == "__main__":
    main()
