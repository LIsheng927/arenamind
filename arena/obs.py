"""Observations: what each player is allowed to know, as a fixed-size vector.

Players never see the full game state. Each observation is written from one
player's point of view:

    * positions and velocities of other things are given relative to the
      player, as (forward, right) distances along its facing direction, so the
      same situation looks the same whichever way the player faces;
    * the opponent is only shown where it was last seen, plus how long ago,
      because the policy has no memory of its own;
    * bullets are only shown if the player has line of sight to them;
    * every number is divided by a typical maximum, so all inputs have a
      similar size (roughly -1 to 1).

Layout (block sizes come from ObsConfig, 49 numbers in total):

    self      10  hp, shoot cooldown, dash cooldown, velocity (fwd, right),
                  x, y, cos(facing), sin(facing), time left
    opponent   8  visible, position (fwd, right), distance, hp,
                  velocity (fwd, right), time since last seen
    rays      16  distance to the nearest wall or cover, starting straight
                  ahead and going clockwise
    bullets   15  the 3 nearest visible enemy bullets, each:
                  present, position (fwd, right), velocity (fwd, right)
"""

from __future__ import annotations

import numpy as np

from . import geometry as geo

UNSEEN_SCALE = 10.0  # seconds; "time since last seen" reaches 1.0 here and stays there


def to_local(vec: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split world vectors into (forward, right) parts for a player facing (cos, sin)."""
    x, y = vec[..., 0], vec[..., 1]
    return x * cos + y * sin, -x * sin + y * cos


def observe(env) -> np.ndarray:
    """Observations of every player in every game, shape (num_envs, 2, obs_dim), float32."""
    cfg, oc = env.cfg, env.cfg.obs
    n, size = env.num_envs, cfg.arena.size
    max_speed = cfg.agent.speed * cfg.dash.speed_multiplier
    cos, sin = np.cos(env.angle), np.sin(env.angle)  # (n, 2)
    me = env.pos  # (n, 2, 2)

    # --- self ----------------------------------------------------------------
    vf, vr = to_local(env.vel, cos, sin)
    time_left = (cfg.max_steps - env.t) / cfg.max_steps
    own = np.stack([
        env.hp / cfg.agent.max_hp,
        env.shoot_cd / cfg.shoot_cooldown_steps,
        env.dash_cd / cfg.dash_cooldown_steps,
        vf / max_speed,
        vr / max_speed,
        me[..., 0] / size,
        me[..., 1] / size,
        cos,
        sin,
        np.broadcast_to(time_left[:, None], (n, 2)),
    ], axis=-1)

    # --- opponent ------------------------------------------------------------
    # last_seen is the opponent's current position whenever it is visible.
    visible = np.broadcast_to(env.visible[:, None], (n, 2))
    rel = env.last_seen - me
    pf, pr = to_local(rel, cos, sin)
    ovf, ovr = to_local(env.vel[:, ::-1], cos, sin)  # opponent's velocity, in my frame
    unseen_seconds = env.unseen * cfg.dt
    # The opponent's HP stays known while it hides: you know how often you hit it.
    other = np.stack([
        visible.astype(np.float64),
        pf / size,
        pr / size,
        np.linalg.norm(rel, axis=-1) / size,
        env.hp[:, ::-1] / cfg.agent.max_hp,
        np.where(visible, ovf, 0.0) / max_speed,
        np.where(visible, ovr, 0.0) / max_speed,
        np.minimum(unseen_seconds / UNSEEN_SCALE, 1.0),
    ], axis=-1)

    # --- range sensors ---------------------------------------------------------
    offsets = np.arange(oc.n_rays) * (2 * np.pi / oc.n_rays)  # 0 = straight ahead, clockwise
    ray_angle = env.angle[..., None] + offsets  # (n, 2, rays)
    ray_dir = np.stack([np.cos(ray_angle), np.sin(ray_angle)], axis=-1)
    ray_origin = np.broadcast_to(me[:, :, None, :], ray_dir.shape)
    rays = geo.raycast(ray_origin, ray_dir, env.obstacles, size, oc.ray_range) / oc.ray_range

    # --- enemy bullets ---------------------------------------------------------------
    # Player p's enemy bullets are the ones owned by player 1 - p.
    b_alive = env.bullet_alive[:, ::-1]  # (n, 2, slots)
    b_pos = env.bullet_pos[:, ::-1]
    b_vel = env.bullet_vel[:, ::-1] / cfg.dt  # per second
    origin = np.broadcast_to(me[:, :, None, :], b_pos.shape)
    seen = b_alive & geo.line_of_sight(origin, b_pos, env.obstacles)
    dist = np.where(seen, np.linalg.norm(b_pos - origin, axis=-1), np.inf)
    nearest = np.argsort(dist, axis=-1, kind="stable")[..., :oc.tracked_bullets]
    seen = np.take_along_axis(seen, nearest, axis=-1)
    b_rel = np.take_along_axis(b_pos - origin, nearest[..., None], axis=2)
    b_vel = np.take_along_axis(b_vel, nearest[..., None], axis=2)
    bf, br = to_local(b_rel, cos[..., None], sin[..., None])
    bvf, bvr = to_local(b_vel, cos[..., None], sin[..., None])
    speed = cfg.weapon.bullet_speed
    bullets = np.stack([np.ones_like(bf), bf / size, br / size, bvf / speed, bvr / speed], axis=-1)
    bullets = (bullets * seen[..., None]).reshape(n, 2, -1)  # empty slots are all zeros

    obs = np.concatenate([own, other, rays, bullets], axis=-1).astype(np.float32)
    assert obs.shape[-1] == oc.obs_dim
    return obs


# Index ranges of each block, for tests and debugging tools.
def blocks(cfg) -> dict[str, slice]:
    oc = cfg.obs
    sizes = {"self": oc.self_dim, "opponent": oc.opponent_dim, "rays": oc.n_rays,
             "bullets": oc.tracked_bullets * oc.bullet_dim}
    out, start = {}, 0
    for name, k in sizes.items():
        out[name] = slice(start, start + k)
        start += k
    return out
