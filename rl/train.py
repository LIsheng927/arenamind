"""Train a policy with PPO.

    python -m rl.train --name first --steps 20000000
    python -m rl.train --name smoke --steps 300000     # quick check that it runs

Progress is printed every 10 iterations and written to runs/<name>/log.csv;
the model is saved to runs/<name>/model.pt.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .ppo import PPOConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="run", help="folder name under runs/")
    parser.add_argument("--steps", type=int, default=PPOConfig.total_steps)
    parser.add_argument("--num-envs", type=int, default=PPOConfig.num_envs)
    parser.add_argument("--lr", type=float, default=PPOConfig.learning_rate)
    parser.add_argument("--entropy", type=float, default=PPOConfig.entropy_coef)
    parser.add_argument("--opponent", default=PPOConfig.opponent)
    parser.add_argument("--seed", type=int, default=PPOConfig.seed)
    parser.add_argument("--device", default=PPOConfig.device, choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    cfg = PPOConfig(total_steps=args.steps, num_envs=args.num_envs, learning_rate=args.lr,
                    entropy_coef=args.entropy, opponent=args.opponent, seed=args.seed,
                    device=args.device)
    train(cfg, Path("runs") / args.name)


if __name__ == "__main__":
    main()
