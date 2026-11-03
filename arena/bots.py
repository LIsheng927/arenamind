"""Hand-written opponents, and a helper to play many games between two of them.

A bot controls one player (0 or 1) in every game of an ArenaEnv:

    actions[:, p] = bot.act(env)     # (num_envs, 4) ints
    ...env.step(actions)...
    bot.on_done(env, done)           # forget per-game memory of finished games

RandomBot presses random buttons; it is what an untrained policy looks like.
RuleBot is classic hand-written game AI: chase, aim, strafe, dodge. It is the
first opponent a trained policy has to beat, and a fixed yardstick later on.
It only uses what a player could know: its own state, whether it can see the
opponent, where it last saw it, and bullets in its line of sight.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from . import geometry as geo
from .config import DEFAULT_CONFIG, GameConfig
from .env import ArenaEnv, wrap_angle

# Where the rule bot looks when it has lost track of the opponent.
# Point-symmetric like the map, so neither side gets a better search route.
PATROL_POINTS = np.array([[4, 4], [16, 16], [4, 16], [16, 4], [10, 4], [10, 16], [3, 9], [17, 11]],
                         dtype=np.float64)

# Move indices (relative to facing), matching MOVE_DIRECTIONS in config.py.
STAY, FORWARD, FWD_RIGHT, RIGHT, BACK_RIGHT, BACK, BACK_LEFT, LEFT, FWD_LEFT = range(9)


class RandomBot:
    def __init__(self, env: ArenaEnv, player: int, seed: int | None = None) -> None:
        self.player = player
        self.sizes = env.cfg.action_sizes
        self.rng = np.random.default_rng(seed)

    def act(self, env: ArenaEnv) -> np.ndarray:
        n = env.num_envs
        return np.stack([self.rng.integers(0, k, size=n) for k in self.sizes], axis=-1)

    def on_done(self, env: ArenaEnv, done: np.ndarray) -> None:
        pass


class IdleBot:
    """Stands still and never shoots: a sitting target, for checking that a
    trained policy can find and finish an opponent that does not come to it."""

    def __init__(self, env: ArenaEnv, player: int, seed: int | None = None) -> None:
        self.player = player

    def act(self, env: ArenaEnv) -> np.ndarray:
        actions = np.zeros((env.num_envs, 4), dtype=np.int64)
        actions[:, 1] = 2  # no turn
        return actions

    def on_done(self, env: ArenaEnv, done: np.ndarray) -> None:
        pass


def turn_towards(diff: np.ndarray, turn_degrees: tuple[float, ...]) -> np.ndarray:
    """Pick the turn choice that brings the facing closest to `diff` radians away."""
    turns = np.radians(np.array(turn_degrees))
    return np.argmin(np.abs(diff[..., None] - turns), axis=-1)


def move_towards(rel_angle: np.ndarray) -> np.ndarray:
    """Pick the move whose direction is closest to `rel_angle` (0 = straight ahead, clockwise)."""
    sector = np.round(rel_angle / (np.pi / 4)).astype(np.int64) % 8  # 0 ahead, 2 right, 4 back
    return 1 + sector


class RuleBot:
    REACTION_STEPS = 3  # must have seen the opponent this long before shooting
    AIM_TOLERANCE = np.radians(10)
    TOO_CLOSE, TOO_FAR = 4.0, 9.0  # keeps this distance band while fighting
    ARRIVED = 0.8  # close enough to a search point to call it checked
    DODGE_LOOKAHEAD = 2  # steps of bullet flight to check for danger
    DODGE_MARGIN = 0.4  # extra clearance when deciding a bullet is dangerous

    def __init__(self, env: ArenaEnv, player: int, seed: int | None = None) -> None:
        self.player = player
        self.cfg = env.cfg
        self.rng = np.random.default_rng(seed)
        n = env.num_envs
        self.goal = np.zeros((n, 2))
        self.patrolling = np.zeros(n, dtype=bool)  # last known spot checked, now searching
        self.seen_for = np.zeros(n, dtype=np.int64)
        self.side = np.ones(n, dtype=np.int64)  # strafe direction: +1 right, -1 left
        self.side_timer = np.zeros(n, dtype=np.int64)
        self.stuck = np.zeros(n, dtype=np.int64)
        self.prev_pos = env.pos[:, player].copy()
        self._forget(np.ones(n, dtype=bool), env)

    def on_done(self, env: ArenaEnv, done: np.ndarray) -> None:
        self._forget(done, env)

    def _forget(self, mask: np.ndarray, env: ArenaEnv) -> None:
        idx = np.flatnonzero(mask)
        self.goal[idx] = PATROL_POINTS[self.rng.integers(len(PATROL_POINTS), size=idx.size)]
        self.patrolling[idx] = False
        self.seen_for[idx] = 0
        self.side[idx] = self.rng.choice([-1, 1], size=idx.size)
        self.side_timer[idx] = self.rng.integers(8, 15, size=idx.size)
        self.stuck[idx] = 0
        self.prev_pos[idx] = env.pos[idx, self.player]

    def act(self, env: ArenaEnv) -> np.ndarray:
        cfg, p, n = self.cfg, self.player, env.num_envs
        me, angle = env.pos[:, p], env.angle[:, p]
        visible = env.visible
        last_seen = env.last_seen[:, p]

        # --- memory: what am I heading for? ---------------------------------
        self.seen_for = np.where(visible, self.seen_for + 1, 0)
        self.patrolling &= ~visible
        moved = np.linalg.norm(me - self.prev_pos, axis=-1)
        self.prev_pos = me.copy()
        self.stuck = np.where(~visible & (moved < 0.1), self.stuck + 1, 0)
        reached_last_seen = ~visible & ~self.patrolling & \
            (np.linalg.norm(last_seen - me, axis=-1) < self.ARRIVED)
        reached_goal = self.patrolling & (np.linalg.norm(self.goal - me, axis=-1) < self.ARRIVED)
        new_goal = reached_last_seen | reached_goal | (self.stuck >= 5)
        self.patrolling |= new_goal
        k = np.flatnonzero(new_goal)
        self.goal[k] = PATROL_POINTS[self.rng.integers(len(PATROL_POINTS), size=k.size)]
        self.stuck[k] = 0

        target = np.where(visible[:, None], env.pos[:, 1 - p],
                          np.where(self.patrolling[:, None], self.goal, last_seen))
        rel = target - me
        dist = np.linalg.norm(rel, axis=-1)
        want = np.arctan2(rel[:, 1], rel[:, 0])

        # --- aim -----------------------------------------------------------------
        turn = turn_towards(wrap_angle(want - angle), cfg.agent.turn_degrees)
        new_angle = angle + np.radians(np.array(cfg.agent.turn_degrees))[turn]
        aim_error = np.abs(wrap_angle(want - new_angle))
        shoot = visible & (aim_error < self.AIM_TOLERANCE) & (self.seen_for >= self.REACTION_STEPS)

        # --- move ----------------------------------------------------------------
        # Fighting: strafe sideways, switching sides now and then, and keep a
        # middle distance. Otherwise: walk towards the target.
        self.side_timer = np.where(visible, self.side_timer - 1, self.side_timer)
        flip = visible & (self.side_timer <= 0)
        self.side = np.where(flip, -self.side, self.side)
        self.side_timer = np.where(flip, self.rng.integers(8, 15, size=n), self.side_timer)
        right = self.side > 0
        strafe = np.where(right, RIGHT, LEFT)
        strafe = np.where(dist < self.TOO_CLOSE, np.where(right, BACK_RIGHT, BACK_LEFT), strafe)
        strafe = np.where(dist > self.TOO_FAR, np.where(right, FWD_RIGHT, FWD_LEFT), strafe)
        walk = move_towards(wrap_angle(want - new_angle))
        move = np.where(visible, strafe, walk)

        # --- dodge -----------------------------------------------------------------
        # A visible enemy bullet that would touch me within the next two steps:
        # dash sideways if the dash is ready, otherwise just sidestep.
        danger = self._danger(env, me)
        move = np.where(danger, np.where(right, RIGHT, LEFT), move)
        dash = danger & (env.dash_cd[:, p] == 0)

        return np.stack([move, turn, shoot.astype(np.int64), dash.astype(np.int64)], axis=-1)

    def _danger(self, env: ArenaEnv, me: np.ndarray) -> np.ndarray:
        cfg, q = self.cfg, 1 - self.player
        alive = env.bullet_alive[:, q]  # enemy bullets, (n, slots)
        if not alive.any():
            return np.zeros(env.num_envs, dtype=bool)
        p0 = env.bullet_pos[:, q]
        p1 = p0 + env.bullet_vel[:, q] * self.DODGE_LOOKAHEAD
        centre = me[:, None, :]
        reach = cfg.agent.radius + cfg.weapon.bullet_radius + self.DODGE_MARGIN
        incoming = np.isfinite(geo.segment_circle_entry(p0, p1, centre, reach))
        in_sight = geo.line_of_sight(np.broadcast_to(centre, p0.shape), p0, env.obstacles)
        return (alive & incoming & in_sight).any(axis=-1)


BOTS: dict[str, Callable[[ArenaEnv, int, int | None], object]] = {
    "random": RandomBot,
    "rule": RuleBot,
    "idle": IdleBot,
}


def play_matches(blue: str, orange: str, num_games: int, seed: int = 0,
                 cfg: GameConfig = DEFAULT_CONFIG) -> dict:
    """Play num_games games (all at once) and report the results from blue's side."""
    env = ArenaEnv(num_games, cfg, seed=seed)
    bots = [BOTS[blue](env, 0, seed + 1), BOTS[orange](env, 1, seed + 2)]
    finished = np.zeros(num_games, dtype=bool)
    outcome = np.zeros(num_games, dtype=np.int64)
    length = np.zeros(num_games, dtype=np.int64)
    hits = np.zeros((num_games, 2), dtype=np.int64)
    while not finished.all():
        actions = np.stack([b.act(env) for b in bots], axis=1)
        _, done, info = env.step(actions)
        playing = ~finished
        hits[playing] += np.rint(info["damage_dealt"][playing] / cfg.weapon.damage).astype(np.int64)
        new = done & playing
        outcome[new] = info["outcome"][new, 0]
        length[new] = info["episode_length"][new]
        finished |= done
        for b in bots:
            b.on_done(env, done)
    return {
        "games": num_games,
        "wins": int((outcome > 0).sum()),
        "losses": int((outcome < 0).sum()),
        "draws": int((outcome == 0).sum()),
        "hits_per_game": float(hits[:, 0].mean()),
        "hit_by_per_game": float(hits[:, 1].mean()),
        "mean_length": float(length.mean()),
    }
