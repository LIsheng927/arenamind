"""Play a trained policy against a bot and report the results.

    python -m rl.evaluate runs/first/model.pt --games 1000
    python -m rl.evaluate runs/first/model.pt --side orange --opponent idle
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
    while not finished.all():
        actions = np.stack([b.act(env) for b in bots], axis=1)
        _, done, info = env.step(actions)
        new = done & ~finished
        outcome[new] = info["outcome"][new, side]
        finished |= done
        for b in bots:
            b.on_done(env, done)
    return {"games": games, "wins": int((outcome > 0).sum()),
            "losses": int((outcome < 0).sum()), "draws": int((outcome == 0).sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint")
    parser.add_argument("--opponent", choices=sorted(BOTS), default="rule")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--greedy", action="store_true",
                        help="always take the most likely action instead of sampling")
    parser.add_argument("--side", choices=["blue", "orange"], default="blue")
    args = parser.parse_args()
    side = 0 if args.side == "blue" else 1
    r = evaluate(args.checkpoint, args.opponent, args.games, args.seed, args.greedy, side)
    print(f"as {args.side} vs {args.opponent}: {r['wins']} wins, {r['losses']} losses, {r['draws']} draws "
          f"out of {r['games']} ({r['wins'] / r['games']:.1%} win rate)")


if __name__ == "__main__":
    main()
