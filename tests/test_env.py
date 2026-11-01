"""Tests for the batched arena simulation."""

import math

import numpy as np
import pytest

from arena.config import GameConfig
from arena.env import ArenaEnv, wrap_angle

CFG = GameConfig()
S = CFG.arena.size
R = CFG.agent.radius

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


# ---- reset ------------------------------------------------------------------

def test_reset_starts_every_game_fresh():
    env = ArenaEnv(8, seed=1)
    spawns = np.array(CFG.arena.spawns)
    assert np.all(env.t == 0)
    assert np.all(env.hp == 100)
    assert not env.bullet_alive.any()
    assert np.all(np.abs(env.pos - spawns) <= CFG.arena.spawn_jitter)
    assert np.all((env.angle >= -math.pi) & (env.angle < math.pi))
    assert np.allclose(env.last_seen, spawns[::-1])  # each knows where the other spawned
    assert np.all(env.unseen == 0)
    assert not env.visible.any()  # the centre block hides the spawns from each other


def test_same_seed_same_game():
    a, b = ArenaEnv(4, seed=3), ArenaEnv(4, seed=3)
    rng = np.random.default_rng(0)
    for _ in range(200):
        acts = np.stack([rng.integers(0, n, size=(4, 2)) for n in CFG.action_sizes], axis=-1)
        ra, da, _ = a.step(acts)
        rb, db, _ = b.step(acts)
        assert np.array_equal(ra, rb) and np.array_equal(da, db)
    assert np.array_equal(a.pos, b.pos) and np.array_equal(a.hp, b.hp)


# ---- turning and moving --------------------------------------------------------

def test_turning():
    env = env_with(*FAR, a0=0.0, a1=170.0)
    env.step(act((0, 4, 0, 0), (0, 4, 0, 0)))  # both +30 degrees
    assert np.degrees(env.angle[0, 0]) == pytest.approx(30.0)
    assert np.degrees(env.angle[0, 1]) == pytest.approx(-160.0)  # 200 wraps to -160
    env.step(act((0, 0, 0, 0)))  # -30 degrees
    assert np.degrees(env.angle[0, 0]) == pytest.approx(0.0, abs=1e-9)


def test_wrap_angle():
    assert wrap_angle(np.array([math.pi, -math.pi, 3 * math.pi])) == pytest.approx([-math.pi] * 3)


def test_moves_are_relative_to_facing():
    # Facing down (90 degrees): forward is +y, and the right hand points to -x.
    env = env_with((10.0, 3.0), (3.0, 17.0), a0=90.0)
    env.step(act(FORWARD))
    assert env.pos[0, 0] == pytest.approx([10.0, 3.4])
    assert env.vel[0, 0] == pytest.approx([0.0, 4.0])  # units per second
    env.step(act(RIGHT))
    assert env.pos[0, 0] == pytest.approx([9.6, 3.4])


def test_walking_back_into_cover_stops_flush_against_it():
    # At (8.6, 10) facing left, stepping back 0.4 would reach x = 9.0, inside
    # the centre block's margin; the player stops at 9 - 0.5 = 8.5.
    env = env_with((8.6, 10.0), (17.0, 3.0), a0=180.0)
    env.step(act(BACK))
    assert env.pos[0, 0] == pytest.approx([8.5, 10.0])


def test_players_cannot_overlap():
    env = env_with((10.0, 3.0), (11.0, 3.0), a0=0.0, a1=180.0)
    for _ in range(5):
        env.step(act(FORWARD, FORWARD))  # walk into each other
    assert math.dist(env.pos[0, 0], env.pos[0, 1]) == pytest.approx(2 * R)


def test_dash_is_three_times_faster_for_three_steps():
    env = env_with((5.0, 3.0), (3.0, 17.0), a0=0.0)
    xs = []
    for _ in range(5):
        env.step(act(DASH_FORWARD))
        xs.append(env.pos[0, 0, 0])
    assert np.diff([5.0] + xs) == pytest.approx([1.2, 1.2, 1.2, 0.4, 0.4])
    assert env.dash_cd[0, 0] == 30 - 5  # cooldown counts from the start of the dash


def test_dash_cooldown():
    env = env_with((5.0, 3.0), (3.0, 17.0), a0=0.0)
    env.step(act(DASH_FORWARD))
    for _ in range(29):
        env.step(act((0, 2, 0, 1)))  # keep asking to dash while standing still
    assert env.dash_cd[0, 0] == 0 and env.dash_left[0, 0] == 0
    x = env.pos[0, 0, 0]
    env.step(act(DASH_FORWARD))  # 30 steps later it works again
    assert env.pos[0, 0, 0] - x == pytest.approx(1.2)


def test_dash_cannot_tunnel_through_thin_cover():
    # Just below the 1-unit-thick bar (y in [11, 12]) and dashing straight up.
    # One 1.2 jump would put the centre inside the bar, closer to its far side.
    env = env_with((6.0, 12.5), (17.0, 3.0), a0=-90.0)
    env.step(act(DASH_FORWARD))
    assert env.pos[0, 0] == pytest.approx([6.0, 12.5])


# ---- bullets ---------------------------------------------------------------------

def test_bullet_appears_at_muzzle_then_flies_ten_steps():
    env = env_with(*FAR, a0=0.0)
    env.step(act(SHOOT))
    k = np.flatnonzero(env.bullet_alive[0, 0])
    assert k.size == 1
    k = k[0]
    assert env.bullet_pos[0, 0, k] == pytest.approx([3.6, 3.0])  # 0.5 body + 0.1 bullet
    assert env.bullet_life[0, 0, k] == 10
    for i in range(1, 11):
        env.step(act())
        if i < 10:
            assert env.bullet_pos[0, 0, k] == pytest.approx([3.6 + 1.5 * i, 3.0])
            assert env.bullet_life[0, 0, k] == 10 - i
    assert not env.bullet_alive.any()  # flew 15 units and vanished


def test_shots_are_four_steps_apart_and_never_need_a_fourth_slot():
    env = env_with(*FAR, a0=0.0)
    fired = []
    for i in range(20):
        env.step(act(SHOOT))
        # A bullet still has its full 10 steps of life only on the step it was fired.
        if (env.bullet_alive[0, 0] & (env.bullet_life[0, 0] == 10)).any():
            fired.append(i)
        assert env.bullet_alive[0, 0].sum() <= 3
    assert fired == [0, 4, 8, 12, 16]


def test_bullet_hits_opponent_and_rewards_are_zero_sum():
    # Shooter at x=3, target at x=8. The bullet starts at 3.6 and touches the
    # target (radius 0.5 + bullet 0.1) once it passes x = 7.4: on its 3rd move.
    env = env_with((3.0, 3.0), (8.0, 3.0), a0=0.0)
    rewards = []
    for i in range(4):
        r, done, info = env.step(act(SHOOT if i == 0 else STAY))
        rewards.append(r[0].copy())
    assert rewards[:3] == [pytest.approx([0, 0])] * 3
    assert rewards[3] == pytest.approx([0.2, -0.2])
    assert env.hp[0] == pytest.approx([100, 80])
    assert not env.bullet_alive.any()


def test_bullet_stops_at_thin_cover():
    # The bar y in [11, 12] is between them. The bullet starts at y = 13.4 and
    # would jump to 11.9 in one move; it must still be stopped by the bar.
    env = env_with((6.0, 14.0), (6.0, 9.5), a0=-90.0)
    env.step(act(SHOOT))
    for _ in range(10):
        env.step(act())
    assert env.hp[0, 1] == 100
    assert not env.bullet_alive.any()


def test_shooting_point_blank_into_cover_wastes_the_shot():
    env = env_with((8.5, 10.0), (17.0, 3.0), a0=0.0)  # touching the centre block, facing it
    env.step(act(SHOOT))
    assert not env.bullet_alive.any()
    assert env.shoot_cd[0, 0] == 3  # cooldown was used (4, then ticked once)


# ---- end of game and rewards ----------------------------------------------------

def test_kill_wins_and_game_restarts():
    env = env_with((3.0, 3.0), (8.0, 3.0), a0=0.0, hp=(100.0, 20.0))
    for i in range(4):
        r, done, info = env.step(act(SHOOT if i == 0 else STAY))
    assert done[0]
    assert info["outcome"][0].tolist() == [1, -1]
    assert r[0] == pytest.approx([1.2, -1.2])  # +-0.2 for the hit, +-1 for the result
    assert info["episode_length"][0] == 4
    assert env.t[0] == 0 and env.hp[0].tolist() == [100, 100]  # already reset


def test_both_dying_in_the_same_step_is_a_draw():
    env = env_with((3.0, 3.0), (8.0, 3.0), a0=0.0, a1=180.0, hp=(20.0, 20.0))
    for i in range(4):
        r, done, info = env.step(act(SHOOT, SHOOT) if i == 0 else act())
    assert done[0]
    assert info["outcome"][0].tolist() == [0, 0]
    assert r[0] == pytest.approx([0.0, 0.0])  # +0.2 dealt - 0.2 taken + 0 for a draw


@pytest.mark.parametrize("hp, outcome", [((60.0, 40.0), [1, -1]), ((40.0, 60.0), [-1, 1]),
                                         ((60.0, 60.0), [0, 0])])
def test_time_limit_compares_hp(hp, outcome):
    env = env_with(*FAR, hp=hp)
    env.t[0] = CFG.max_steps - 1
    r, done, info = env.step(act())
    assert done[0]
    assert info["outcome"][0].tolist() == outcome
    assert info["reward_outcome"][0].tolist() == outcome
    assert info["reward_damage"][0].tolist() == [0, 0]


def test_reset_only_touches_finished_games():
    env = ArenaEnv(2, seed=0)
    env.t[:] = [CFG.max_steps - 1, 10]
    _, done, _ = env.step(act(n=2))
    assert done.tolist() == [True, False]
    assert env.t.tolist() == [0, 11]


# ---- what each player has seen ----------------------------------------------------

def test_sight_memory():
    env = env_with((5.5, 15.0), (12.0, 15.0))
    env.step(act())
    assert env.visible[0]
    assert env.last_seen[0, 0] == pytest.approx([12.0, 15.0])
    assert env.unseen[0].tolist() == [0, 0]

    env.pos[0, 1] = [15.33, 6.67]  # now behind the centre block
    for i in range(1, 4):
        env.step(act())
        assert not env.visible[0]
        assert env.unseen[0].tolist() == [i, i]
    assert env.last_seen[0, 0] == pytest.approx([12.0, 15.0])  # still the old spot


# ---- whole-game checks ---------------------------------------------------------------

def random_actions(rng, n):
    return np.stack([rng.integers(0, size, size=(n, 2)) for size in CFG.action_sizes], axis=-1)


def test_random_play_keeps_the_game_valid():
    n = 256
    env = ArenaEnv(n, seed=0)
    rng = np.random.default_rng(1)
    rects = env.obstacles
    finished = 0
    for _ in range(400):
        r, done, info = env.step(random_actions(rng, n))
        finished += done.sum()
        assert np.isfinite(r).all() and np.isfinite(env.pos).all() and np.isfinite(env.angle).all()
        assert np.allclose(r.sum(axis=1), 0.0)  # zero-sum every step
        assert np.all((env.pos >= R - 1e-9) & (env.pos <= S - R + 1e-9))
        # Distance from every player to every rectangle is at least the radius.
        p = env.pos[:, :, None, :]
        nearest = np.clip(p, rects[:, :2], rects[:, 2:])
        assert (np.linalg.norm(p - nearest, axis=-1) >= R - 1e-6).all()
        assert np.all((env.hp >= 0) & (env.hp <= 100))
        assert np.all(env.t < CFG.max_steps)
        assert np.all(env.bullet_alive.sum(axis=-1) <= CFG.weapon.max_bullets)
        assert np.all(info["episode_length"] <= CFG.max_steps)
        assert set(np.unique(info["outcome"])) <= {-1, 0, 1}
    assert finished > 0


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


def test_mirrored_game_plays_out_mirrored():
    """Neither player is favoured by the order the code handles them in.

    Rotating the map 180 degrees maps it onto itself, so swapping the players
    and mirroring everything must give exactly mirrored results.
    """
    n = 64
    env = ArenaEnv(n, seed=5)
    rng = np.random.default_rng(6)
    for _ in range(30):  # get some bullets and damage going first
        env.step(random_actions(rng, n))
    mirror = mirrored_copy(env)
    active = np.ones(n, dtype=bool)  # games compared until they first end
    hits = 0
    while active.any():
        a = random_actions(rng, n)
        r, done, info = env.step(a)
        rm, dm, _ = mirror.step(a[:, ::-1].copy())
        assert np.array_equal(done[active], dm[active])
        assert np.allclose(r[active], rm[active][:, ::-1], atol=1e-9)
        hits += (info["damage_dealt"][active] > 0).sum()
        active &= ~done
        assert np.allclose(env.pos[active], S - mirror.pos[active][:, ::-1], atol=1e-6)
        assert np.array_equal(env.hp[active], mirror.hp[active][:, ::-1])
    assert hits > 0  # the comparison covered bullets actually hitting players


def test_rejects_bad_actions():
    env = ArenaEnv(1, seed=0)
    with pytest.raises(ValueError, match="shape"):
        env.step(np.zeros((2, 4), dtype=np.int64))
    with pytest.raises(ValueError, match="move"):
        env.step(act((9, 2, 0, 0)))
    with pytest.raises(ValueError, match="integers"):
        env.step(act().astype(float))
