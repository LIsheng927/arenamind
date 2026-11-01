"""Record a game between two bots as a GIF.

    python -m tools.replay --blue rule --orange rule --out runs/rule_vs_rule.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from arena.bots import BOTS
from arena.env import ArenaEnv
from arena.render import render


def record(blue: str, orange: str, seed: int, max_steps: int, scale: int,
           rays: bool) -> list:
    env = ArenaEnv(1, seed=seed, auto_reset=False)  # keep the final moment on screen
    bots = [BOTS[blue](env, 0, seed + 1), BOTS[orange](env, 1, seed + 2)]
    ray_player = 0 if rays else None
    frames = [render(env, scale=scale, rays_for=ray_player)]
    for _ in range(max_steps):
        actions = np.stack([b.act(env) for b in bots], axis=1)
        _, done, info = env.step(actions)
        status = ""
        if done[0]:
            status = {1: "blue wins", -1: "orange wins", 0: "draw"}[int(info["outcome"][0, 0])]
        frames.append(render(env, scale=scale, rays_for=ray_player, status=status))
        if done[0]:
            frames += [frames[-1]] * 15  # hold the final frame for 1.5 s
            break
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--blue", choices=sorted(BOTS), default="rule")
    parser.add_argument("--orange", choices=sorted(BOTS), default="rule")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--scale", type=int, default=20, help="pixels per arena unit")
    parser.add_argument("--rays", action="store_true", help="draw blue's range rays")
    parser.add_argument("--out", type=Path, default=Path("runs/replay.gif"))
    args = parser.parse_args()

    frames = record(args.blue, args.orange, args.seed, args.max_steps, args.scale, args.rays)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=100, loop=0,
                   optimize=True)
    print(f"wrote {args.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
