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

CamperBot, RusherBot, KiterBot and FlankerBot aim and shoot like RuleBot but
move in other ways (hold a spot, charge in, keep away, come from the side). No
training run uses them: they show whether a policy learned to fight or only
learned the habits of the opponents it trained against.
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
        # Three parts, so that other bots can reuse the first two and only
        # change how they move: what do I know, where do I aim, where do I go.
        s = self._sense(env)
        turn, shoot = self._aim(s)
        move, dash = self._move(env, s)
        return np.stack([move, turn, shoot.astype(np.int64), dash.astype(np.int64)], axis=-1)

    def _sense(self, env: ArenaEnv) -> dict:
        """Update memory and pick the point to look at; returns what the other parts need."""
        p = self.player
        me, angle = env.pos[:, p], env.angle[:, p]
        visible = env.visible
        last_seen = env.last_seen[:, p]

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

        target = self._target(env, visible, last_seen)
        rel = target - me
        return {"me": me, "angle": angle, "visible": visible,
                "dist": np.linalg.norm(rel, axis=-1), "want": np.arctan2(rel[:, 1], rel[:, 0])}

    def _target(self, env: ArenaEnv, visible: np.ndarray, last_seen: np.ndarray) -> np.ndarray:
        """The opponent while visible; otherwise where it was last seen, then patrol points."""
        return np.where(visible[:, None], env.pos[:, 1 - self.player],
                        np.where(self.patrolling[:, None], self.goal, last_seen))

    def _aim(self, s: dict) -> tuple[np.ndarray, np.ndarray]:
        """Turn towards the target; shoot once it has been in view long enough and is lined up."""
        turns = self.cfg.agent.turn_degrees
        turn = turn_towards(wrap_angle(s["want"] - s["angle"]), turns)
        s["new_angle"] = s["angle"] + np.radians(np.array(turns))[turn]
        aim_error = np.abs(wrap_angle(s["want"] - s["new_angle"]))
        shoot = s["visible"] & (aim_error < self.AIM_TOLERANCE) & (self.seen_for >= self.REACTION_STEPS)
        return turn, shoot

    def _move(self, env: ArenaEnv, s: dict) -> tuple[np.ndarray, np.ndarray]:
        visible, dist = s["visible"], s["dist"]
        # Fighting: strafe sideways, switching sides now and then, and keep a
        # middle distance. Otherwise: walk towards the target.
        self.side_timer = np.where(visible, self.side_timer - 1, self.side_timer)
        flip = visible & (self.side_timer <= 0)
        self.side = np.where(flip, -self.side, self.side)
        self.side_timer = np.where(flip, self.rng.integers(8, 15, size=env.num_envs), self.side_timer)
        right = self.side > 0
        strafe = np.where(right, RIGHT, LEFT)
        strafe = np.where(dist < self.TOO_CLOSE, np.where(right, BACK_RIGHT, BACK_LEFT), strafe)
        strafe = np.where(dist > self.TOO_FAR, np.where(right, FWD_RIGHT, FWD_LEFT), strafe)
        walk = move_towards(wrap_angle(s["want"] - s["new_angle"]))
        move = np.where(visible, strafe, walk)

        # A visible enemy bullet that would touch me within the next two steps:
        # dash sideways if the dash is ready, otherwise just sidestep.
        danger = self._danger(env, s["me"])
        move = np.where(danger, np.where(right, RIGHT, LEFT), move)
        dash = danger & (env.dash_cd[:, self.player] == 0)
        return move, dash

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


# --- Opponents with other play styles ------------------------------------------
# They see, aim and shoot exactly like RuleBot and only move differently, so a
# policy that beats RuleBot but loses to one of these has learned that bot's
# habits rather than how to fight.

class CamperBot(RuleBot):
    """Holds its spawn point: turns towards where it last saw the opponent and
    shoots, sidesteps incoming bullets, then walks back to its spot."""

    HOME_RADIUS = 0.8  # close enough to its spot to stand still

    def _forget(self, mask: np.ndarray, env: ArenaEnv) -> None:
        super()._forget(mask, env)
        if not hasattr(self, "home"):
            self.home = np.zeros((env.num_envs, 2))
        self.home[mask] = env.pos[mask, self.player]

    def _target(self, env: ArenaEnv, visible: np.ndarray, last_seen: np.ndarray) -> np.ndarray:
        # Never patrols: keeps watching the last known position.
        return np.where(visible[:, None], env.pos[:, 1 - self.player], last_seen)

    def _move(self, env: ArenaEnv, s: dict) -> tuple[np.ndarray, np.ndarray]:
        to_home = self.home - s["me"]
        away = np.linalg.norm(to_home, axis=-1) > self.HOME_RADIUS
        home_dir = np.arctan2(to_home[:, 1], to_home[:, 0])
        move = np.where(away, move_towards(wrap_angle(home_dir - s["new_angle"])), STAY)
        danger = self._danger(env, s["me"])
        move = np.where(danger, np.where(self.side > 0, RIGHT, LEFT), move)
        dash = danger & (env.dash_cd[:, self.player] == 0)
        return move, dash


class RusherBot(RuleBot):
    """Charges straight at the opponent and dashes to close the distance.
    Never strafes or keeps its distance, but still sidesteps bullets."""

    DASH_RANGE = 6.0  # dash towards a visible opponent further away than this

    def _move(self, env: ArenaEnv, s: dict) -> tuple[np.ndarray, np.ndarray]:
        move = move_towards(wrap_angle(s["want"] - s["new_angle"]))
        danger = self._danger(env, s["me"])
        move = np.where(danger, np.where(self.side > 0, RIGHT, LEFT), move)
        charge = s["visible"] & (s["dist"] > self.DASH_RANGE)
        dash = (danger | charge) & (env.dash_cd[:, self.player] == 0)
        return move, dash


class KiterBot(RuleBot):
    """Fights from long range: backs away when the opponent comes closer than
    10 units and only moves in when it is further than 14."""

    TOO_CLOSE, TOO_FAR = 10.0, 14.0


class FlankerBot(RuleBot):
    """Comes at the opponent from the side instead of straight on, then circles
    it in one direction at mid range without ever switching sides.

    Kept for a final test: it was written before any policy played it, and no
    training run or design decision has looked at how policies do against it."""

    TOO_CLOSE, TOO_FAR = 5.0, 8.0
    SWING = 5.0  # how far to the side of the straight line it walks

    def _target(self, env: ArenaEnv, visible: np.ndarray, last_seen: np.ndarray) -> np.ndarray:
        target = super()._target(env, visible, last_seen)
        rel = target - env.pos[:, self.player]
        dist = np.linalg.norm(rel, axis=-1, keepdims=True)
        sideways = np.stack([-rel[:, 1], rel[:, 0]], axis=-1) / np.maximum(dist, 1e-6)
        flank = target + sideways * self.side[:, None] * np.minimum(self.SWING, dist / 2)
        flank = np.clip(flank, 1.0, self.cfg.arena.size - 1.0)
        return np.where(visible[:, None], target, flank)

    def _move(self, env: ArenaEnv, s: dict) -> tuple[np.ndarray, np.ndarray]:
        self.side_timer[:] = 10**6  # keep circling the same way
        return super()._move(env, s)


BOTS: dict[str, Callable[[ArenaEnv, int, int | None], object]] = {
    "random": RandomBot,
    "rule": RuleBot,
    "idle": IdleBot,
    "camper": CamperBot,
    "rusher": RusherBot,
    "kiter": KiterBot,
    "flanker": FlankerBot,
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
