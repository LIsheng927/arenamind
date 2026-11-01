"""Game configuration for ArenaMind.

Every tunable number in the game lives in this file, so balancing the game
(bullet speed, map layout, rewards, ...) never requires touching simulation
code.

Coordinate conventions (same as screen coordinates):
    * x grows to the right, y grows downward; the arena is [0, size] x [0, size].
    * Angles are radians in code (this file uses degrees only for readability).
      Angle 0 faces +x, and a positive angle turns clockwise on screen.
    * facing vector     = ( cos a, sin a)
      right-hand vector = (-sin a, cos a)

Time is stored in two forms: seconds (for humans editing this file) and whole
simulation steps (for the code). All timers in the simulation count steps as
integers, because repeatedly subtracting 0.1 from a float drifts: 0.4 minus
0.1 four times is 2.8e-17, not 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

Rect = tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)
Vec2 = tuple[float, float]

# The 9 movement choices, in the agent's own frame as (forward, right).
# Diagonals are scaled to length 1 so moving diagonally is not faster.
_D = math.sqrt(0.5)
MOVE_DIRECTIONS: tuple[Vec2, ...] = (
    (0.0, 0.0),  # 0 stay
    (1.0, 0.0),  # 1 forward
    (_D, _D),  # 2 forward-right
    (0.0, 1.0),  # 3 right
    (-_D, _D),  # 4 back-right
    (-1.0, 0.0),  # 5 back
    (-_D, -_D),  # 6 back-left
    (0.0, -1.0),  # 7 left
    (_D, -_D),  # 8 forward-left
)


def seconds_to_steps(seconds: float, dt: float, name: str) -> int:
    """Convert a duration to a whole number of steps, or fail loudly."""
    steps = round(seconds / dt)  # not int(): 0.3 / 0.1 == 2.9999999999999996
    if abs(steps * dt - seconds) > 1e-9:
        raise ValueError(f"{name} = {seconds}s is not a whole number of {dt}s steps")
    return steps


@dataclass(frozen=True)
class ArenaConfig:
    size: float = 20.0
    # Point-symmetric about the centre (10, 10), so neither side has an
    # advantage. Mirror of (x0, y0, x1, y1) is (S-x1, S-y1, S-x0, S-y0).
    obstacles: tuple[Rect, ...] = (
        (9.0, 9.0, 11.0, 11.0),  # centre block
        (4.0, 11.0, 8.0, 12.0),  # lower-left bar
        (12.0, 8.0, 16.0, 9.0),  # upper-right bar (mirror of the one above)
        (6.0, 6.0, 7.0, 7.0),  # upper-left pillar
        (13.0, 13.0, 14.0, 14.0),  # lower-right pillar (mirror of the one above)
    )
    # Player 0 starts bottom-left, player 1 top-right; the centre block
    # blocks the line between them, so nobody can see the other at the start.
    spawns: tuple[Vec2, Vec2] = ((3.0, 17.0), (17.0, 3.0))
    spawn_jitter: float = 0.5  # random offset in [-j, j] on each axis


@dataclass(frozen=True)
class AgentConfig:
    radius: float = 0.5
    max_hp: float = 100.0
    speed: float = 4.0  # units per second
    turn_degrees: tuple[float, ...] = (-30.0, -8.0, 0.0, 8.0, 30.0)  # per step


@dataclass(frozen=True)
class WeaponConfig:
    bullet_speed: float = 15.0  # units per second
    bullet_radius: float = 0.1
    bullet_lifetime: float = 1.0  # seconds, so range = 15 units
    damage: float = 20.0  # 5 hits from full HP
    cooldown: float = 0.4  # seconds between shots
    max_bullets: int = 3  # bullet slots per player


@dataclass(frozen=True)
class DashConfig:
    speed_multiplier: float = 3.0
    duration: float = 0.3  # seconds
    cooldown: float = 3.0  # seconds, counted from the start of the dash


@dataclass(frozen=True)
class ObsConfig:
    n_rays: int = 16
    ray_range: float = 10.0
    tracked_bullets: int = 3  # nearest enemy bullets the agent can see

    # Sizes of each block of the observation vector (see docs/env_spec.md).
    self_dim: int = 10  # hp, 2 cooldowns, 2 velocity, x, y, cos, sin, time left
    opponent_dim: int = 8  # visible, 2 rel. pos, dist, hp, 2 rel. vel, time unseen
    bullet_dim: int = 5  # present, 2 rel. pos, 2 rel. vel

    @property
    def obs_dim(self) -> int:
        return (
            self.self_dim
            + self.opponent_dim
            + self.n_rays
            + self.tracked_bullets * self.bullet_dim
        )


@dataclass(frozen=True)
class RewardConfig:
    win: float = 1.0
    loss: float = -1.0
    draw: float = 0.0
    # Shaping is zero-sum: what I gain by hitting, you lose by being hit.
    per_damage_dealt: float = 0.01  # 20 damage -> +0.2
    per_damage_taken: float = 0.01  # 20 damage -> -0.2


@dataclass(frozen=True)
class GameConfig:
    dt: float = 0.1  # seconds per step (agents decide 10 times per second)
    max_time: float = 60.0  # then the player with more HP wins
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    weapon: WeaponConfig = field(default_factory=WeaponConfig)
    dash: DashConfig = field(default_factory=DashConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        _validate(self)

    # ---- durations in whole steps -------------------------------------
    @property
    def max_steps(self) -> int:
        return seconds_to_steps(self.max_time, self.dt, "max_time")

    @property
    def shoot_cooldown_steps(self) -> int:
        return seconds_to_steps(self.weapon.cooldown, self.dt, "weapon.cooldown")

    @property
    def bullet_lifetime_steps(self) -> int:
        return seconds_to_steps(self.weapon.bullet_lifetime, self.dt, "weapon.bullet_lifetime")

    @property
    def dash_steps(self) -> int:
        return seconds_to_steps(self.dash.duration, self.dt, "dash.duration")

    @property
    def dash_cooldown_steps(self) -> int:
        return seconds_to_steps(self.dash.cooldown, self.dt, "dash.cooldown")

    # ---- distances per step -------------------------------------------
    @property
    def move_per_step(self) -> float:
        return self.agent.speed * self.dt  # 0.4

    @property
    def bullet_per_step(self) -> float:
        return self.weapon.bullet_speed * self.dt  # 1.5

    @property
    def dash_substeps(self) -> int:
        """A dashing player makes this many normal-length moves per step."""
        return round(self.dash.speed_multiplier)  # 3

    # ---- action space ---------------------------------------------------
    @property
    def action_sizes(self) -> tuple[int, int, int, int]:
        """Number of choices for (move, turn, shoot, dash)."""
        return (len(MOVE_DIRECTIONS), len(self.agent.turn_degrees), 2, 2)


def _mirror_rect(r: Rect, size: float) -> Rect:
    x0, y0, x1, y1 = r
    return (size - x1, size - y1, size - x0, size - y0)


def _close(a: tuple[float, ...], b: tuple[float, ...], tol: float = 1e-9) -> bool:
    return all(abs(p - q) <= tol for p, q in zip(a, b))


def _validate(cfg: GameConfig) -> None:
    """Catch config mistakes at startup instead of as weird training results."""
    a, size = cfg.arena, cfg.arena.size

    # 1. Every duration must be a whole number of steps (raises otherwise).
    _ = (cfg.max_steps, cfg.shoot_cooldown_steps, cfg.bullet_lifetime_steps,
         cfg.dash_steps, cfg.dash_cooldown_steps)

    # 2. Obstacles are proper rectangles inside the arena.
    for r in a.obstacles:
        x0, y0, x1, y1 = r
        if not (0 <= x0 < x1 <= size and 0 <= y0 < y1 <= size):
            raise ValueError(f"obstacle {r} is not a valid rectangle inside the arena")

    # 3. The map is point-symmetric, so both players face the same map.
    for r in a.obstacles:
        m = _mirror_rect(r, size)
        if not any(_close(m, other) for other in a.obstacles):
            raise ValueError(f"obstacle {r} has no mirror image {m}; the map is unfair")
    (sx0, sy0), (sx1, sy1) = a.spawns
    if not _close((size - sx0, size - sy0), (sx1, sy1)):
        raise ValueError("spawn points are not mirror images of each other")

    # 4. A spawned player (plus worst-case jitter) never starts inside a wall.
    worst = cfg.agent.radius + a.spawn_jitter * math.sqrt(2)
    for sx, sy in a.spawns:
        if not (worst <= sx <= size - worst and worst <= sy <= size - worst):
            raise ValueError(f"spawn {(sx, sy)} is too close to the arena edge")
        for x0, y0, x1, y1 in a.obstacles:
            nx, ny = min(max(sx, x0), x1), min(max(sy, y0), y1)  # nearest point
            if math.hypot(sx - nx, sy - ny) < worst:
                raise ValueError(f"spawn {(sx, sy)} is too close to obstacle {(x0, y0, x1, y1)}")

    # 5. A single move is shorter than the player's radius, so a player can never
    #    jump into cover in one move. A dash is split into several normal moves,
    #    so its speed must be a whole multiple of the normal speed.
    if cfg.move_per_step >= cfg.agent.radius:
        raise ValueError(f"moving {cfg.move_per_step} per step is not shorter than "
                         f"the player radius {cfg.agent.radius}")
    m = cfg.dash.speed_multiplier
    if m < 1 or abs(m - round(m)) > 1e-9:
        raise ValueError(f"dash.speed_multiplier must be a whole number >= 1, got {m}")

    # 6. Observations can only show bullets that exist.
    if cfg.obs.tracked_bullets > cfg.weapon.max_bullets:
        raise ValueError("obs.tracked_bullets cannot exceed weapon.max_bullets")

    # 7. Enough bullet slots: firing non-stop must never need a 4th slot.
    needed = math.ceil(cfg.bullet_lifetime_steps / cfg.shoot_cooldown_steps)
    if needed > cfg.weapon.max_bullets:
        raise ValueError(f"non-stop firing keeps {needed} bullets alive, "
                         f"but max_bullets is {cfg.weapon.max_bullets}")


DEFAULT_CONFIG = GameConfig()
