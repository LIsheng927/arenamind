"""Batched simulation of the 1v1 arena.

One ArenaEnv runs N independent games at once. Every piece of game state is a
numpy array whose first axis is the game and whose second axis is the player
(0 or 1), so a single line of numpy updates every game at the same time.

step() advances every game by one tick (cfg.dt seconds) in a fixed order:

    1. read both players' actions
    2. turn
    3. move, then push players out of walls, cover and each other
    4. advance bullets that are already flying (hits and damage)
    5. fire new bullets
    6. tick timers
    7. check whether each game is over
    8. compute rewards
    9. update what each player has seen of the other

Each stage handles both players together before the next stage starts, so
neither player ever acts on fresher information than the other. Finished
games are reset immediately, so callers can keep calling step() forever.
"""

from __future__ import annotations

import numpy as np

from . import geometry as geo
from .config import DEFAULT_CONFIG, MOVE_DIRECTIONS, GameConfig

ACTION_FIELDS = ("move", "turn", "shoot", "dash")


def wrap_angle(a: np.ndarray) -> np.ndarray:
    """Map angles to [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


class ArenaEnv:
    def __init__(self, num_envs: int = 1, cfg: GameConfig = DEFAULT_CONFIG,
                 seed: int | None = None) -> None:
        self.cfg = cfg
        self.num_envs = n = num_envs
        self.rng = np.random.default_rng(seed)
        k = cfg.weapon.max_bullets

        # Fixed for the whole run.
        self.obstacles = np.array(cfg.arena.obstacles, dtype=np.float64)  # (rects, 4)
        self._spawns = np.array(cfg.arena.spawns, dtype=np.float64)  # (2 players, 2)
        self._move_dirs = np.array(MOVE_DIRECTIONS, dtype=np.float64)  # (9, 2) as (forward, right)
        self._turns = np.radians(np.array(cfg.agent.turn_degrees, dtype=np.float64))  # (5,)

        # The game state: everything that changes during a game.
        # Shapes: [game], [game, player], [game, player, xy], [game, owner, slot, ...].
        self.t = np.zeros(n, dtype=np.int64)  # steps since the game started
        self.pos = np.zeros((n, 2, 2))  # player centre (x, y)
        self.angle = np.zeros((n, 2))  # facing, radians in [-pi, pi)
        self.vel = np.zeros((n, 2, 2))  # how far each player moved last step, per second
        self.hp = np.zeros((n, 2))
        self.shoot_cd = np.zeros((n, 2), dtype=np.int64)  # steps until the next shot is allowed
        self.dash_cd = np.zeros((n, 2), dtype=np.int64)  # steps until the next dash is allowed
        self.dash_left = np.zeros((n, 2), dtype=np.int64)  # steps of the current dash still to run
        self.bullet_alive = np.zeros((n, 2, k), dtype=bool)
        self.bullet_pos = np.zeros((n, 2, k, 2))
        self.bullet_vel = np.zeros((n, 2, k, 2))  # displacement per step
        self.bullet_life = np.zeros((n, 2, k), dtype=np.int64)  # steps of flight left
        # What each player knows about the other. Observations only show the
        # opponent while it is visible, so its last known position has to be
        # remembered here.
        self.visible = np.zeros(n, dtype=bool)  # line of sight between the two players
        self.last_seen = np.zeros((n, 2, 2))  # [game, viewer]: opponent's last known position
        self.unseen = np.zeros((n, 2), dtype=np.int64)  # steps since the viewer last saw it

        self.reset()

    # ------------------------------------------------------------------
    # Starting games
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> None:
        """Start a new game in every slot."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_games(np.ones(self.num_envs, dtype=bool))

    def _reset_games(self, mask: np.ndarray) -> None:
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return
        m, a = idx.size, self.cfg.arena
        jitter = self.rng.uniform(-a.spawn_jitter, a.spawn_jitter, size=(m, 2, 2))
        self.pos[idx] = self._spawns + jitter
        self.angle[idx] = self.rng.uniform(-np.pi, np.pi, size=(m, 2))
        self.vel[idx] = 0.0
        self.hp[idx] = self.cfg.agent.max_hp
        self.shoot_cd[idx] = 0
        self.dash_cd[idx] = 0
        self.dash_left[idx] = 0
        self.bullet_alive[idx] = False
        self.bullet_pos[idx] = 0.0
        self.bullet_vel[idx] = 0.0
        self.bullet_life[idx] = 0
        self.t[idx] = 0
        # Spawn points are part of the map, so each player starts out knowing
        # where the opponent began.
        self.last_seen[idx] = self._spawns[::-1]
        self.unseen[idx] = 0
        self._update_sight(idx, tick=False)

    # ------------------------------------------------------------------
    # One tick
    # ------------------------------------------------------------------

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        """Advance every game by one tick.

        actions: int array of shape (num_envs, 2, 4) holding
                 (move, turn, shoot, dash) for each player of each game.
        Returns (reward, done, info):
            reward: (num_envs, 2) float
            done:   (num_envs,) bool; games that finished were already reset
            info:   per-step details, recorded before any reset
        """
        cfg = self.cfg
        n = self.num_envs

        # 1. Read actions.
        actions = np.asarray(actions)
        if actions.shape != (n, 2, 4):
            raise ValueError(f"actions must have shape {(n, 2, 4)}, got {actions.shape}")
        if not np.issubdtype(actions.dtype, np.integer):
            raise ValueError(f"actions must be integers, got {actions.dtype}")
        for k, (name, size) in enumerate(zip(ACTION_FIELDS, cfg.action_sizes)):
            a = actions[..., k]
            if a.min() < 0 or a.max() >= size:
                raise ValueError(f"{name} actions must be in [0, {size - 1}]")
        move, turn, shoot, dash = (actions[..., k] for k in range(4))
        hp_before = self.hp.copy()

        # 2. Turn.
        self.angle = wrap_angle(self.angle + self._turns[turn])

        # 3. Move.
        self._move(move, dash.astype(bool))

        # 4. Bullets already in flight.
        self._advance_bullets()

        # 5. New bullets.
        self._fire(shoot.astype(bool))

        # 6. Timers.
        self.shoot_cd = np.maximum(self.shoot_cd - 1, 0)
        self.dash_cd = np.maximum(self.dash_cd - 1, 0)
        self.dash_left = np.maximum(self.dash_left - 1, 0)
        self.t += 1

        # 7. Game over? A player at 0 HP loses; at the time limit, more HP wins.
        #    Equal HP (including both players dying in the same step) is a draw.
        done = (self.hp <= 0).any(axis=1) | (self.t >= cfg.max_steps)
        lead = np.sign(self.hp[:, 0] - self.hp[:, 1]).astype(np.int64)
        outcome = np.stack([lead, -lead], axis=1) * done[:, None]  # +1 win, -1 loss, 0 draw/ongoing

        # 8. Rewards. HP actually lost (not raw bullet damage) keeps this
        #    zero-sum even when two bullets hit a player with 20 HP left.
        rc = cfg.reward
        taken = hp_before - self.hp
        dealt = taken[:, ::-1]
        r_damage = rc.per_damage_dealt * dealt - rc.per_damage_taken * taken
        r_outcome = np.where(outcome > 0, rc.win, np.where(outcome < 0, rc.loss, rc.draw))
        r_outcome = r_outcome * done[:, None]
        reward = r_damage + r_outcome

        # 9. What each player can see now.
        self._update_sight(np.arange(n), tick=True)

        info = {
            "reward_damage": r_damage,
            "reward_outcome": r_outcome,
            "damage_dealt": dealt,
            "outcome": outcome,
            "episode_length": np.where(done, self.t, 0),
        }
        if done.any():
            self._reset_games(done)
        return reward, done, info

    # ------------------------------------------------------------------
    # Stages of a step
    # ------------------------------------------------------------------

    def _move(self, move: np.ndarray, dash_requested: np.ndarray) -> None:
        cfg = self.cfg

        # A dash starts only when both its cooldown and any current dash are over.
        start = dash_requested & (self.dash_cd == 0) & (self.dash_left == 0)
        self.dash_left = np.where(start, cfg.dash_steps, self.dash_left)
        self.dash_cd = np.where(start, cfg.dash_cooldown_steps, self.dash_cd)
        moves = np.where(self.dash_left > 0, cfg.dash_substeps, 1)  # normal-length moves this step

        # Turn the chosen (forward, right) direction into a world direction.
        fwd, right = self._move_dirs[move][..., 0], self._move_dirs[move][..., 1]
        c, s = np.cos(self.angle), np.sin(self.angle)
        world = np.stack([fwd * c - right * s, fwd * s + right * c], axis=-1)
        one_move = world * cfg.move_per_step

        # Dashes are split into normal-length moves with collisions resolved
        # after each one. Every move is shorter than the radius, so a player can
        # never end up inside a 1-unit-thick wall and be pushed out the far side.
        start_pos = self.pos.copy()
        for i in range(int(moves.max())):
            active = (i < moves)[..., None]
            self.pos = self.pos + np.where(active, one_move, 0.0)
            self._resolve_collisions()
        self.vel = (self.pos - start_pos) / cfg.dt

    def _resolve_collisions(self) -> None:
        r = self.cfg.agent.radius
        # Players may not overlap: push both apart by half the overlap each.
        d = self.pos[:, 1] - self.pos[:, 0]
        dist = np.linalg.norm(d, axis=-1)
        apart = dist > 1e-9
        normal = np.where(apart[:, None], d / np.where(apart, dist, 1.0)[:, None], [1.0, 0.0])
        push = (np.maximum(2 * r - dist, 0.0) / 2)[:, None] * normal
        self.pos[:, 0] -= push
        self.pos[:, 1] += push
        # Walls and cover always win over the player-player push.
        self.pos = geo.push_out_of_rects(self.pos, r, self.obstacles)
        self.pos = geo.clamp_to_arena(self.pos, r, self.cfg.arena.size)

    def _advance_bullets(self) -> None:
        if not self.bullet_alive.any():
            return
        cfg = self.cfg
        step_len = cfg.bullet_per_step
        p0 = self.bullet_pos
        p1 = p0 + self.bullet_vel

        # How far along this step's path each bullet first meets cover, a wall,
        # or the opponent's body. Checking the whole path, not just the end
        # point, stops fast bullets from skipping over thin cover or a player.
        cover = geo.segment_rects_entry(p0, p1, self.obstacles)
        wall = geo.ray_arena_exit(p0, self.bullet_vel / step_len, cfg.arena.size)
        blocked = np.minimum(cover, wall)
        opponent = self.pos[:, ::-1][:, :, None, :]  # bullets of owner p fly at player 1-p
        body = geo.segment_circle_entry(p0, p1, opponent,
                                        cfg.agent.radius + cfg.weapon.bullet_radius)

        alive = self.bullet_alive
        hit = alive & (body <= step_len) & (body <= blocked)
        stopped = alive & ~hit & (blocked <= step_len)
        flying = alive & ~hit & ~stopped

        hits_by_owner = hit.sum(axis=-1)  # (games, owner)
        self.hp = np.maximum(self.hp - cfg.weapon.damage * hits_by_owner[:, ::-1], 0.0)

        self.bullet_pos = np.where(flying[..., None], p1, p0)
        self.bullet_life = np.where(flying, self.bullet_life - 1, self.bullet_life)
        self.bullet_alive = flying & (self.bullet_life > 0)

    def _fire(self, want: np.ndarray) -> None:
        cfg = self.cfg
        free = ~self.bullet_alive
        attempt = want & (self.shoot_cd == 0) & free.any(axis=-1)
        if not attempt.any():
            return

        # The bullet appears just outside the shooter's body. It starts flying
        # on the next step, so the opponent gets at least one step to react.
        facing = np.stack([np.cos(self.angle), np.sin(self.angle)], axis=-1)
        muzzle = self.pos + facing * (cfg.agent.radius + cfg.weapon.bullet_radius)
        size = cfg.arena.size
        in_arena = ((muzzle >= 0) & (muzzle <= size)).all(axis=-1)
        # Firing point-blank into a wall uses up the shot but creates no bullet.
        spawn = attempt & in_arena & ~self._in_cover(muzzle)
        self.shoot_cd = np.where(attempt, cfg.shoot_cooldown_steps, self.shoot_cd)

        g, p = np.nonzero(spawn)
        slot = np.argmax(free[g, p], axis=-1)  # first free slot
        self.bullet_alive[g, p, slot] = True
        self.bullet_pos[g, p, slot] = muzzle[g, p]
        self.bullet_vel[g, p, slot] = facing[g, p] * cfg.bullet_per_step
        self.bullet_life[g, p, slot] = cfg.bullet_lifetime_steps

    def _update_sight(self, idx: np.ndarray, tick: bool) -> None:
        pos = self.pos[idx]
        vis = geo.line_of_sight(pos[:, 0], pos[:, 1], self.obstacles)
        self.visible[idx] = vis
        opponent = pos[:, ::-1]  # player p looks at player 1-p
        self.last_seen[idx] = np.where(vis[:, None, None], opponent, self.last_seen[idx])
        self.unseen[idx] = np.where(vis[:, None], 0, self.unseen[idx] + int(tick))

    def _in_cover(self, points: np.ndarray) -> np.ndarray:
        x, y = points[..., 0, None], points[..., 1, None]
        r = self.obstacles
        return ((x >= r[:, 0]) & (x <= r[:, 2]) & (y >= r[:, 1]) & (y <= r[:, 3])).any(axis=-1)
