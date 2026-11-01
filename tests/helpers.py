"""Shared helpers for building small, hand-made game situations in tests."""

import math

import numpy as np

from arena.config import GameConfig
from arena.env import ArenaEnv, wrap_angle

CFG = GameConfig()
S = CFG.arena.size

# Action indices: move 0 stay, 1 forward, 3 right, 5 back; turn 2 = no turn.
STAY = (0, 2, 0, 0)
SHOOT = (0, 2, 1, 0)
FORWARD = (1, 2, 0, 0)
BACK = (5, 2, 0, 0)
RIGHT = (3, 2, 0, 0)
DASH_FORWARD = (1, 2, 0, 1)


def act(p0=STAY, p1=STAY, n=1):
    a = np.zeros((n, 2, 4), dtype=np.int64)
    a[:, 0] = p0
    a[:, 1] = p1
    return a


def env_with(p0, p1, a0=0.0, a1=0.0, hp=(100.0, 100.0), n=1, seed=0):
    """A fresh env with game 0 set up by hand. Angles are in degrees."""
    env = ArenaEnv(n, seed=seed)
    env.pos[0] = [p0, p1]
    env.angle[0] = np.radians([a0, a1])
    env.hp[0] = hp
    env._update_sight(np.array([0]), tick=False)
    return env


FAR = ((3.0, 3.0), (17.0, 17.0))  # two spots far apart with open space around


def random_actions(rng, n):
    return np.stack([rng.integers(0, size, size=(n, 2)) for size in CFG.action_sizes], axis=-1)


def mirrored_copy(env):
    """The same games with the map rotated 180 degrees and the players swapped."""
    m = ArenaEnv(env.num_envs, seed=0)
    swap = lambda a: a[:, ::-1].copy()  # noqa: E731
    m.t = env.t.copy()
    m.pos = S - swap(env.pos)
    m.angle = wrap_angle(swap(env.angle) + math.pi)
    m.vel = -swap(env.vel)
    m.hp = swap(env.hp)
    m.shoot_cd, m.dash_cd, m.dash_left = swap(env.shoot_cd), swap(env.dash_cd), swap(env.dash_left)
    m.bullet_alive, m.bullet_life = swap(env.bullet_alive), swap(env.bullet_life)
    m.bullet_pos = np.where(m.bullet_alive[..., None], S - swap(env.bullet_pos), 0.0)
    m.bullet_vel = -swap(env.bullet_vel)
    m.visible = env.visible.copy()
    m.last_seen = S - swap(env.last_seen)
    m.unseen = swap(env.unseen)
    return m
