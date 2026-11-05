"""Checks for the difficulty settings that do not need PyTorch."""

import math

import numpy as np
import pytest

from arena.config import GameConfig
from arena.env import ArenaEnv
from rl.difficulty import (OPPONENT_POSITION, OPPONENT_VELOCITY, Difficulty, DifficultyBot, Handicap,
                           rotate)

CFG = GameConfig()


def test_settings_are_parsed_and_labelled():
    knobs = Difficulty.parse("delay=3,aim=8")
    assert knobs == Difficulty(delay=3, aim=8.0)
    assert knobs.label() == "delay=3,aim=8"
    assert Difficulty.parse(knobs.label()) == knobs
    assert Difficulty().label() == "full"
    assert Difficulty.parse("temp=1.5,every=2") == Difficulty(temperature=1.5, every=2)


@pytest.mark.parametrize("text", ["speed=2", "delay", "every=0", "delay=-1"])
def test_bad_settings_are_refused(text):
    with pytest.raises(ValueError):
        Difficulty.parse(text)


def test_rotate_turns_vectors():
    f, r = rotate(np.array([1.0]), np.array([0.0]), np.array([math.pi / 2]))
    assert np.allclose([f[0], r[0]], [0.0, 1.0])


def test_full_strength_changes_nothing():
    h = Handicap(Difficulty(), 4, CFG.dt, seed=0)
    obs = np.random.default_rng(0).normal(size=(4, CFG.obs.obs_dim)).astype(np.float32)
    assert np.array_equal(h.view(obs), obs)
    new = np.arange(16).reshape(4, 4)
    assert np.array_equal(h.choose(new), new)


def test_delay_shows_an_older_observation_and_starts_afresh():
    h = Handicap(Difficulty(delay=2), 2, CFG.dt)
    seen = []
    for t in range(5):
        obs = np.full((2, CFG.obs.obs_dim), float(t), dtype=np.float32)
        seen.append(h.view(obs)[:, 0].tolist())
        h.choose(np.zeros((2, 4), dtype=np.int64))
        if t == 2:
            h.on_done(np.array([False, True]))  # game 1 ends after step 2
    # Game 0 sees two steps late from the start (the first steps repeat the present).
    assert [s[0] for s in seen] == [0, 0, 0, 1, 2]
    # Game 1 restarts at step 3 and has no older past than that.
    assert [s[1] for s in seen] == [0, 0, 0, 3, 3]


def test_every_repeats_the_last_action_between_decisions():
    h = Handicap(Difficulty(every=3), 1, CFG.dt)
    picked = []
    for t in range(7):
        h.view(np.zeros((1, CFG.obs.obs_dim), dtype=np.float32))
        picked.append(int(h.choose(np.full((1, 4), t))[0, 0]))
    assert picked == [0, 0, 0, 3, 3, 3, 6]


def test_aim_error_turns_the_opponent_but_keeps_its_distance():
    n = 4000
    h = Handicap(Difficulty(aim=10), n, CFG.dt, seed=1)
    obs = np.zeros((n, CFG.obs.obs_dim), dtype=np.float32)
    obs[:, OPPONENT_POSITION[0]] = 0.5  # straight ahead
    obs[:, OPPONENT_VELOCITY[1]] = 0.2
    for _ in range(20):
        seen = h.view(obs)
    f, r = seen[:, OPPONENT_POSITION[0]], seen[:, OPPONENT_POSITION[1]]
    assert np.allclose(np.hypot(f, r), 0.5, atol=1e-5)
    error = np.degrees(np.arctan2(r, f))
    assert abs(error.std() - 10) < 0.6 and abs(error.mean()) < 0.6
    assert (seen[:, :OPPONENT_POSITION[0]] == 0).all()  # nothing else is touched


def test_aim_error_drifts_slowly():
    h = Handicap(Difficulty(aim=10), 2000, CFG.dt, seed=2)
    before = h.error.copy()
    h.view(np.zeros((2000, CFG.obs.obs_dim), dtype=np.float32))
    assert np.corrcoef(before, h.error)[0, 1] > 0.75  # one step later it is still mostly the same


def test_bot_plays_valid_actions_with_a_stand_in_policy():
    """The bot's own logic, with the network replaced by a fixed choice."""
    env = ArenaEnv(8, seed=3)
    bot = DifficultyBot.__new__(DifficultyBot)
    bot.player, bot.knobs = 1, Difficulty(delay=2, aim=5, every=2)
    bot.handicap = Handicap(bot.knobs, env.num_envs, env.cfg.dt, seed=3)
    bot.sample = lambda obs: np.tile([1, 3, 1, 0], (len(obs), 1))
    for _ in range(30):
        actions = np.zeros((8, 2, 4), dtype=np.int64)
        actions[:, 1] = bot.act(env)
        _, done, _ = env.step(actions)
        bot.on_done(env, done)
    assert (actions[:, 1] == [1, 3, 1, 0]).all()
