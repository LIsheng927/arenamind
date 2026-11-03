"""Watch two bots play, one game after another.

    python -m tools.viewer --blue rule --orange random

Keys: Space pause, Right arrow one step while paused, Up/Down speed,
R toggle blue's range rays, Esc quit.
"""

from __future__ import annotations

import argparse

import numpy as np
import pygame

from arena.env import ArenaEnv
from tools.bot_spec import HELP, make_bot
from tools.window import open_window, show

RESULT = {1: "blue wins", -1: "orange wins", 0: "draw"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--blue", default="rule", help=HELP)
    parser.add_argument("--orange", default="rule", help=HELP)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=int, default=32, help="pixels per arena unit")
    parser.add_argument("--speed", type=float, default=10.0, help="game steps per second")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after this many frames")
    args = parser.parse_args()

    env = ArenaEnv(1, seed=args.seed, auto_reset=False)
    bots = [make_bot(args.blue, env, 0, args.seed + 1), make_bot(args.orange, env, 1, args.seed + 2)]
    screen = open_window(env, args.scale, f"ArenaMind: {args.blue} vs {args.orange}")
    clock = pygame.time.Clock()

    score = {1: 0, -1: 0, 0: 0}
    speed, paused, rays, pause_left = args.speed, False, False, 0.0
    last_result, frames, budget = "", 0, 0.0
    running = True
    while running:
        step_once = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT:
                    step_once = True
                elif event.key == pygame.K_UP:
                    speed = min(speed * 2, 640.0)
                elif event.key == pygame.K_DOWN:
                    speed = max(speed / 2, 1.25)
                elif event.key == pygame.K_r:
                    rays = not rays

        dt = clock.tick(60) / 1000.0
        if pause_left > 0:  # show the end of a game for a moment
            pause_left -= dt
            if pause_left <= 0:
                env.reset()
                for b in bots:
                    b.on_done(env, np.ones(1, dtype=bool))
        else:
            budget = 0.0 if paused else min(budget + dt * speed, 50.0)
            steps = 1 if (paused and step_once) else int(budget)
            budget -= 0 if paused else steps
            for _ in range(steps):
                actions = np.stack([b.act(env) for b in bots], axis=1)
                _, done, info = env.step(actions)
                if done[0]:
                    outcome = int(info["outcome"][0, 0])
                    score[outcome] += 1
                    last_result = RESULT[outcome]
                    pause_left = 1.5
                    break

        status = (f"[{score[1]}-{score[-1]}-{score[0]}]  x{speed / 10:g}"
                  + ("  paused" if paused else "") + (f"  {last_result}" if pause_left > 0 else ""))
        show(screen, env, args.scale, rays_for=0 if rays else None, status=status)

        frames += 1
        if args.max_frames and frames >= args.max_frames:
            running = False

    pygame.quit()
    games = sum(score.values())
    print(f"blue {args.blue} vs orange {args.orange}: {score[1]} wins, {score[-1]} losses, "
          f"{score[0]} draws in {games} games")


if __name__ == "__main__":
    main()
