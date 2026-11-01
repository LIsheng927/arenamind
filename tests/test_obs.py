"""Observation tests, mostly worked examples with easy numbers."""

import math

import numpy as np
import pytest

from arena.config import GameConfig
from arena.env import ArenaEnv
from arena.obs import blocks, observe
from tests.helpers import act, env_with, mirrored_copy, random_actions

CFG = GameConfig()
B = blocks(CFG)
SELF, OPP, RAYS, BUL = B["self"], B["opponent"], B["rays"], B["bullets"]
S = CFG.arena.size


def opp(obs, g=0, p=0):
    visible, fwd, right, dist, hp, vf, vr, unseen = obs[g, p, OPP]
    return dict(visible=visible, fwd=fwd * S, right=right * S, dist=dist * S, hp=hp,
                unseen_s=unseen * 10.0)


def test_shape_dtype_and_range():
    env = ArenaEnv(32, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(200):
        env.step(random_actions(rng, 32))
        obs = observe(env)
        assert obs.shape == (32, 2, CFG.obs.obs_dim) and obs.dtype == np.float32
        assert np.isfinite(obs).all()
        assert np.abs(obs).max() <= 1.5  # distances across the diagonal reach 28.3 / 20


def test_opponent_straight_ahead_whichever_way_i_face():
    # Facing right with the opponent 6.5 to the right, or facing up with the
    # opponent 6.5 above: both are "6.5 straight ahead".
    a = observe(env_with((3.0, 15.0), (9.5, 15.0), a0=0.0))
    b = observe(env_with((3.0, 15.0), (3.0, 8.5), a0=-90.0))
    for obs in (a, b):
        o = opp(obs)
        assert o["visible"] == 1
        assert o["fwd"] == pytest.approx(6.5, abs=1e-5)
        assert o["right"] == pytest.approx(0.0, abs=1e-5)
    assert np.allclose(a[0, 0, OPP], b[0, 0, OPP], atol=1e-6)


def test_cover_between_us_means_not_visible():
    # Looking straight up from (5.5, 15), the lower-left bar is in the way.
    assert opp(observe(env_with((5.5, 15.0), (5.5, 8.5), a0=-90.0)))["visible"] == 0


def test_facing_down_an_opponent_to_the_east_is_on_my_left():
    o = opp(observe(env_with((5.5, 15.0), (12.0, 15.0), a0=90.0)))
    assert o["fwd"] == pytest.approx(0.0, abs=1e-5)
    assert o["right"] == pytest.approx(-6.5, abs=1e-5)


def test_hidden_opponent_shows_last_seen_spot_and_time():
    env = env_with((5.5, 15.0), (12.0, 15.0), a0=0.0)
    env.step(act())
    env.pos[0, 1] = [15.33, 6.67]  # behind the centre block
    for _ in range(3):
        env.step(act())
    o = opp(observe(env))
    assert o["visible"] == 0
    assert o["fwd"] == pytest.approx(6.5, abs=1e-5)  # still pointing at (12, 15)
    assert o["unseen_s"] == pytest.approx(0.3, abs=1e-5)


def test_rays_start_straight_ahead_and_go_clockwise():
    obs = observe(env_with((5.5, 15.0), (17.0, 3.0), a0=-90.0))  # facing up
    rays = obs[0, 0, RAYS] * CFG.obs.ray_range
    assert rays[0] == pytest.approx(3.0, abs=1e-5)  # up: the lower-left bar
    assert rays[4] == pytest.approx(10.0, abs=1e-5)  # right: open, capped at 10
    assert rays[8] == pytest.approx(5.0, abs=1e-5)  # down: the bottom wall
    assert rays[12] == pytest.approx(5.5, abs=1e-5)  # left: the left wall


def test_incoming_bullet_is_ahead_and_closing():
    env = env_with((3.0, 3.0), (8.0, 3.0), a0=0.0, a1=180.0)
    env.step(act(p1=(0, 2, 1, 0)))  # orange fires at blue
    present, fwd, right, vf, vr = observe(env)[0, 0, BUL][:5]
    assert present == 1
    assert fwd * S == pytest.approx(4.4, abs=1e-5)  # bullet at x = 7.4, blue at x = 3
    assert right == pytest.approx(0.0, abs=1e-6)
    assert vf == pytest.approx(-1.0, abs=1e-6)  # coming straight at me at full speed
    assert observe(env)[0, 0, BUL][5:].tolist() == [0.0] * 10  # other slots empty


def test_bullets_behind_cover_are_not_shown():
    env = env_with((6.0, 14.0), (6.0, 9.5), a0=-90.0, a1=90.0)
    env.step(act(p1=(0, 2, 1, 0)))  # orange fires down at the bar between them
    assert env.bullet_alive.any()
    assert observe(env)[0, 0, BUL].tolist() == [0.0] * 15


def test_observations_do_not_depend_on_which_side_you_are():
    """Mirror the map and swap players: everything relative stays identical."""
    env = ArenaEnv(16, seed=2)
    rng = np.random.default_rng(3)
    for _ in range(40):
        env.step(random_actions(rng, 16))
    a, b = observe(env), observe(mirrored_copy(env))[:, ::-1]
    absolute = [5, 6, 7, 8]  # x, y, cos, sin are on the map, so they mirror
    relative = [i for i in range(CFG.obs.obs_dim) if i not in absolute]
    assert np.allclose(a[..., relative], b[..., relative], atol=1e-5)
    assert np.allclose(a[..., [5, 6]], 1 - b[..., [5, 6]], atol=1e-5)
    assert np.allclose(a[..., [7, 8]], -b[..., [7, 8]], atol=1e-5)


def test_time_left_counts_down():
    env = ArenaEnv(1, seed=0)
    assert observe(env)[0, 0, SELF][9] == pytest.approx(1.0)
    for _ in range(150):
        env.step(act())
    assert observe(env)[0, 0, SELF][9] == pytest.approx(0.75)
    assert math.isclose(observe(env)[0, 1, SELF][9], 0.75, rel_tol=1e-6)
