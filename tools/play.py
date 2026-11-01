"""Play blue yourself against a bot.

    python -m tools.play --opponent rule
    python -m tools.play --record recordings/me.npz   # save your games

Keys: W A S D move (screen directions), mouse aims, left click or Space
shoots, Shift or right click dashes, Esc quits.

Your input goes through exactly the same action choices a policy has: you
can turn at most 30 degrees per step and move in 8 directions relative to
where you face, so recorded games are valid demonstrations for imitation
learning.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pygame

from arena.bots import BOTS, move_towards, turn_towards
from arena.env import ArenaEnv, wrap_angle
from arena.obs import observe
from tools.window import open_window, show

RESULT = {1: "you win", -1: "you lose", 0: "draw"}


def human_action(env, mouse_xy, keys, buttons, scale: int) -> np.ndarray:
    """Turn keyboard and mouse state into one (move, turn, shoot, dash) action."""
    cfg = env.cfg
    pos, angle = env.pos[0, 0], env.angle[0, 0]
    aim = np.arctan2(mouse_xy[1] / scale - pos[1], mouse_xy[0] / scale - pos[0])
    turn = int(turn_towards(np.array(wrap_angle(aim - angle)), cfg.agent.turn_degrees))
    new_angle = angle + np.radians(cfg.agent.turn_degrees[turn])

    dx = float(keys[pygame.K_d]) - float(keys[pygame.K_a])
    dy = float(keys[pygame.K_s]) - float(keys[pygame.K_w])
    move = 0
    if dx or dy:
        move = int(move_towards(np.array(wrap_angle(np.arctan2(dy, dx) - new_angle))))

    shoot = int(buttons[0] or keys[pygame.K_SPACE])
    dash = int(buttons[2] or keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])
    return np.array([move, turn, shoot, dash])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--opponent", choices=sorted(BOTS), default="rule")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=int, default=32, help="pixels per arena unit")
    parser.add_argument("--record", type=Path, help="save your observations and actions here")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after this many frames")
    args = parser.parse_args()

    env = ArenaEnv(1, seed=args.seed, auto_reset=False)
    bot = BOTS[args.opponent](env, 1, args.seed + 1)
    screen = open_window(env, args.scale, f"ArenaMind: you vs {args.opponent}")
    clock = pygame.time.Clock()
    step_seconds = env.cfg.dt  # the game runs in real time: 10 steps per second

    log = {"obs": [], "action": [], "reward": [], "done": []}
    score = {1: 0, -1: 0, 0: 0}
    budget, pause_left, last_result, frames = 0.0, 0.0, "", 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN
                                             and event.key == pygame.K_ESCAPE):
                running = False

        dt = clock.tick(60) / 1000.0
        if pause_left > 0:
            pause_left -= dt
            if pause_left <= 0:
                env.reset()
                bot.on_done(env, np.ones(1, dtype=bool))
        else:
            budget += dt
            if budget >= step_seconds:
                budget = 0.0
                mine = human_action(env, pygame.mouse.get_pos(), pygame.key.get_pressed(),
                                    pygame.mouse.get_pressed(), args.scale)
                obs = observe(env)[0, 0]
                actions = np.stack([mine[None], bot.act(env)], axis=1)
                reward, done, info = env.step(actions)
                log["obs"].append(obs)
                log["action"].append(mine)
                log["reward"].append(reward[0, 0])
                log["done"].append(done[0])
                if done[0]:
                    outcome = int(info["outcome"][0, 0])
                    score[outcome] += 1
                    last_result = RESULT[outcome]
                    pause_left = 2.0

        status = f"you {score[1]} - {score[-1]} {args.opponent}" + \
            (f"  {last_result}" if pause_left > 0 else "")
        show(screen, env, args.scale, status=status)

        frames += 1
        if args.max_frames and frames >= args.max_frames:
            running = False

    pygame.quit()
    if args.record and log["obs"]:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.record, **{k: np.array(v) for k, v in log.items()})
        print(f"saved {len(log['obs'])} steps to {args.record}")
    print(f"you: {score[1]} wins, {score[-1]} losses, {score[0]} draws")


if __name__ == "__main__":
    main()
