import math

import numpy as np
import pytest

from arena.bots import (
    BACK, FORWARD, FWD_RIGHT, LEFT, RIGHT, RandomBot, RuleBot,
    move_towards, play_matches, turn_towards,
)
from arena.config import GameConfig
from arena.env import ArenaEnv

CFG = GameConfig()


def test_turn_towards_picks_the_closest_turn():
    turns = CFG.agent.turn_degrees  # -30, -8, 0, 8, 30
    diffs = np.radians([20.0, 5.0, 1.0, -12.0, -170.0])
    assert turn_towards(diffs, turns).tolist() == [4, 3, 2, 1, 0]


def test_move_towards_picks_the_closest_direction():
    angles = np.array([0.0, math.pi / 4, math.pi / 2, math.pi, -math.pi / 2, 0.3])
    assert move_towards(angles).tolist() == [FORWARD, FWD_RIGHT, RIGHT, BACK, LEFT, FORWARD]


@pytest.mark.parametrize("bot", [RandomBot, RuleBot])
def test_bots_produce_valid_actions(bot):
    env = ArenaEnv(16, seed=0)
    bots = [bot(env, 0, 1), bot(env, 1, 2)]
    for _ in range(100):
        a = np.stack([b.act(env) for b in bots], axis=1)
        assert a.shape == (16, 2, 4)
        assert ((a >= 0) & (a < np.array(CFG.action_sizes))).all()
        _, done, _ = env.step(a)
        for b in bots:
            b.on_done(env, done)


def test_rule_bot_beats_random_bot():
    r = play_matches("rule", "random", 200, seed=0)
    assert r["wins"] / r["games"] >= 0.95


def test_rule_bot_against_itself_is_even():
    r = play_matches("rule", "rule", 400, seed=0)
    decided = r["wins"] + r["losses"]
    assert 0.4 <= r["wins"] / decided <= 0.6
