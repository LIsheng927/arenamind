"""Play a trained policy against bots and report the results.

    python -m rl.evaluate runs/first/model.pt --games 1000
    python -m rl.evaluate runs/first/model.pt --side orange --opponent idle
    python -m rl.evaluate runs/first/model.pt --opponent rule camper rusher kiter
"""

from __future__ import annotations

import argparse

import numpy as np

from arena.bots import BOTS
from arena.env import ArenaEnv

from .policy_bot import PolicyBot


def evaluate(checkpoint: str, opponent: str, games: int, seed: int, greedy: bool,
             side: int = 0) -> dict:
    """Results from the policy's point of view; side 0 plays blue, side 1 orange."""
    env = ArenaEnv(games, seed=seed)
    policy = PolicyBot(env, side, checkpoint, greedy, seed)
    other = BOTS[opponent](env, 1 - side, seed + 1)
    bots = [policy, other] if side == 0 else [other, policy]
    finished = np.zeros(games, dtype=bool)
    outcome = np.zeros(games, dtype=np.int64)
    length = np.zeros(games, dtype=np.int64)
    hits = np.zeros((games, 2))  # hits landed by blue, by orange
    while not finished.all():
        actions = np.stack([b.act(env) for b in bots], axis=1)
        _, done, info = env.step(actions)
        playing = ~finished
        hits[playing] += np.rint(info["damage_dealt"][playing] / env.cfg.weapon.damage)
        new = done & playing
        outcome[new] = info["outcome"][new, side]
        length[new] = info["episode_length"][new]
        finished |= done
        for b in bots:
            b.on_done(env, done)
    return {"games": games, "wins": int((outcome > 0).sum()),
            "losses": int((outcome < 0).sum()), "draws": int((outcome == 0).sum()),
            "hits": float(hits[:, side].mean()), "hit_by": float(hits[:, 1 - side].mean()),
            "seconds": float(length.mean() * env.cfg.dt)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint")
    parser.add_argument("--opponent", nargs="+", choices=sorted(BOTS), default=["rule"],
                        help="one or more bots to play against")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--greedy", action="store_true",
                        help="always take the most likely action instead of sampling")
    parser.add_argument("--side", choices=["blue", "orange"], default="blue")
    args = parser.parse_args()
    side = 0 if args.side == "blue" else 1
    width = max(len(name) for name in args.opponent)
    for name in args.opponent:
        r = evaluate(args.checkpoint, name, args.games, args.seed, args.greedy, side)
        print(f"as {args.side} vs {name:<{width}}: {r['wins']:>4} wins {r['losses']:>4} losses "
              f"{r['draws']:>4} draws of {r['games']} ({r['wins'] / r['games']:6.1%}) | "
              f"hits {r['hits']:.2f}, hit by {r['hit_by']:.2f}, {r['seconds']:.1f} s per game")


if __name__ == "__main__":
    main()
