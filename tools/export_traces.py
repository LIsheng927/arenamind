"""Record games for checking the Unity (C#) version of the game against this one.

    python -m tools.export_traces    # writes unity/Assets/ArenaMind/Tests/Editor/traces.json

The C# port of arena/ (unity/Assets/ArenaMind/Sim) has to play exactly like
the Python original and show the policy exactly the observations it was
trained on; a policy fed slightly different numbers plays badly without any
error message. So this records a few games between hand-written bots: where
the players started, every action taken, and after every step the game state
and both players' observations. The Unity tests (ArenaMind.Tests) replay the
same actions in C# and compare everything, step by step.

It also records the small pieces the game uses besides the simulation:
wrap_angle, turning an aiming error into a turn choice and a direction into a
move choice (arena/bots.py, used for the human player's input), and which
steps a policy deciding every N steps decides on (rl/difficulty.py).

The layout suits Unity's JsonUtility: one flat array per quantity, steps one
after another (pos holds x0, y0, x1, y1 for step 1, then for step 2, ...).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from arena.bots import BOTS, move_towards, turn_towards
from arena.config import GameConfig
from arena.env import ArenaEnv, wrap_angle
from arena.obs import observe
from rl.difficulty import Difficulty, Handicap

OUT = Path("unity/Assets/ArenaMind/Tests/Editor/traces.json")
# (player 0, player 1, seed): fights with hits and dashes, random play, a chase,
# and a game that runs to the time limit
GAMES = (("rule", "rusher", 0), ("random", "rule", 1), ("kiter", "flanker", 2), ("idle", "camper", 3))
EVERY = (1.0, 1.4, 1.5, 2.2, 2.75, 3.95)  # decision gaps to record
GATE_STEPS, GATE_RESET = 40, 25  # steps per gap, and the step at which a new game starts


def exact(values) -> list[float]:
    """Doubles as they are (Python prints the shortest text that reads back the same number)."""
    return [float(v) for v in np.ravel(values)]


def single(values) -> list[float]:
    """float32 values with 9 significant digits, which is enough to read back the same float32."""
    return [float(f"{v:.9g}") for v in np.ravel(np.asarray(values, dtype=np.float32))]


def whole(values) -> list[int]:
    return [int(v) for v in np.ravel(values)]


def config_numbers(cfg: GameConfig) -> dict:
    """Every setting of arena/config.py, flat, so the C# defaults can be checked against them."""
    a, ag, w, d, o = cfg.arena, cfg.agent, cfg.weapon, cfg.dash, cfg.obs
    return {"dt": cfg.dt, "maxTime": cfg.max_time, "size": a.size,
            "obstacles": exact(a.obstacles), "spawns": exact(a.spawns), "spawnJitter": a.spawn_jitter,
            "radius": ag.radius, "maxHp": ag.max_hp, "speed": ag.speed,
            "turnDegrees": exact(ag.turn_degrees),
            "bulletSpeed": w.bullet_speed, "bulletRadius": w.bullet_radius,
            "bulletLifetime": w.bullet_lifetime, "damage": w.damage, "cooldown": w.cooldown,
            "maxBullets": w.max_bullets, "dashMultiplier": d.speed_multiplier,
            "dashDuration": d.duration, "dashCooldown": d.cooldown, "rays": o.n_rays,
            "rayRange": o.ray_range, "trackedBullets": o.tracked_bullets, "obsDim": o.obs_dim,
            "maxSteps": cfg.max_steps, "actionSizes": list(cfg.action_sizes)}


def record_game(first: str, second: str, seed: int, cfg: GameConfig = GameConfig()) -> dict:
    env = ArenaEnv(1, cfg, seed=seed, auto_reset=False)
    bots = [BOTS[first](env, 0, seed), BOTS[second](env, 1, seed + 100)]
    game = {"name": f"{first} vs {second}", "seed": seed, "steps": 0,
            "startPos": exact(env.pos[0]), "startAngle": exact(env.angle[0]),
            "startObs": single(observe(env)[0])}
    keys = ("actions", "pos", "angle", "vel", "hp", "timers", "bullets", "bulletPos",
            "visible", "unseen", "lastSeen", "done", "outcome", "obs")
    rows: dict[str, list] = {k: [] for k in keys}
    done = np.zeros(1, dtype=bool)
    while not done[0]:
        actions = np.stack([bot.act(env) for bot in bots], axis=1)
        _, done, info = env.step(actions)
        rows["actions"] += whole(actions[0])
        rows["pos"] += exact(env.pos[0])
        rows["angle"] += exact(env.angle[0])
        rows["vel"] += exact(env.vel[0])
        rows["hp"] += exact(env.hp[0])
        rows["timers"] += whole([env.shoot_cd[0], env.dash_cd[0], env.dash_left[0]])
        rows["bullets"] += whole(env.bullet_alive[0])
        rows["bulletPos"] += exact(np.where(env.bullet_alive[0][..., None], env.bullet_pos[0], 0.0))
        rows["visible"] += whole(env.visible[:1])
        rows["unseen"] += whole(env.unseen[0])
        rows["lastSeen"] += exact(env.last_seen[0])
        rows["done"] += whole(done)
        rows["outcome"] += whole(info["outcome"][0, :1])  # +1: player 0 won, -1: lost, 0: draw
        rows["obs"] += single(observe(env)[0])
        game["steps"] += 1
    return {**game, **rows}


def gate_pattern(every: float, dt: float) -> list[int]:
    """1 on the steps a policy deciding every `every` steps decides, with a new game at GATE_RESET."""
    gate = Handicap(Difficulty(every=every), 1, dt)
    out = []
    for step in range(GATE_STEPS):
        if step == GATE_RESET:
            gate.on_done(np.ones(1, dtype=bool))
        out.append(int(gate.deciding()[0]))
        gate.choose(np.full((1, 4), step, dtype=np.int64))
    return out


def controls(cfg: GameConfig, seed: int = 0) -> dict:
    """Inputs and outputs of wrap_angle, turn_towards and move_towards.

    Besides random inputs: whole degrees, where two turn choices can be equally
    close (turn_towards takes the first), and multiples of 1/8 turn, where a
    direction lies halfway between two moves (rounded half to even). These are
    given as whole numbers, so the C# tests work out the very same angles
    instead of reading them back from text.
    """
    rng = np.random.default_rng(seed)
    degrees = np.array([-40, -30, -19, -8, -4, 0, 4, 8, 19, 30, 40])
    eighths = np.arange(-9, 10)
    diffs, angles, wraps = rng.uniform(-3.2, 3.2, 40), rng.uniform(-4, 4, 40), rng.uniform(-20, 20, 30)
    turns = cfg.agent.turn_degrees
    return {"wrapIn": exact(wraps), "wrapOut": exact(wrap_angle(wraps)),
            "turnDegrees": whole(degrees),
            "turnsForDegrees": whole(turn_towards(np.radians(degrees), turns)),
            "turnDiffs": exact(diffs), "turns": whole(turn_towards(diffs, turns)),
            "moveEighths": whole(eighths),
            "movesForEighths": whole(move_towards(np.array([k * math.pi / 8 for k in eighths]))),
            "moveAngles": exact(angles), "moves": whole(move_towards(angles))}


def traces(cfg: GameConfig = GameConfig(), games=GAMES) -> dict:
    return {"config": config_numbers(cfg),
            "games": [record_game(a, b, seed, cfg) for a, b, seed in games],
            "gates": [{"every": e, "newGameAt": GATE_RESET, "decide": gate_pattern(e, cfg.dt)}
                      for e in EVERY],
            "controls": controls(cfg)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    data = traces(games=GAMES)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
    for game in data["games"]:
        outcome = {1: "player 0 won", -1: "player 1 won", 0: "draw"}[game["outcome"][-1]]
        print(f"{game['name']:<18} {game['steps']:>3} steps, {outcome}, "
              f"{sum(game['actions'][3::4])} dashes, hp {game['hp'][-2]:.0f}-{game['hp'][-1]:.0f}")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
