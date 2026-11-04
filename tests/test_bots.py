import math

import numpy as np
import pytest

from arena.bots import (
    BACK, BOTS, FORWARD, FWD_RIGHT, LEFT, RIGHT, CamperBot, IdleBot, KiterBot, RandomBot,
    RuleBot, RusherBot, move_towards, play_matches, turn_towards,
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


@pytest.mark.parametrize("bot", [RandomBot, RuleBot, IdleBot, CamperBot, RusherBot, KiterBot])
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


def test_bot_spec_builds_builtin_bots_and_rejects_unknown_names():
    from tools.bot_spec import make_bot
    env = ArenaEnv(2, seed=0)
    assert isinstance(make_bot("rule", env, 1, 0), RuleBot)
    assert isinstance(make_bot("random", env, 0, 0), RandomBot)
    with pytest.raises(SystemExit):
        make_bot("nonsense", env, 0, 0)


def test_idle_bot_never_moves_or_shoots():
    env = ArenaEnv(4, seed=0)
    bot = IdleBot(env, 1)
    start = env.pos[:, 1].copy()
    for _ in range(50):
        env.step(np.stack([np.tile([0, 2, 0, 0], (4, 1)), bot.act(env)], axis=1))
    assert np.allclose(env.pos[:, 1], start)
    assert not env.bullet_alive[:, 1].any()


def test_rule_bot_finds_and_beats_a_sitting_target():
    r = play_matches("rule", "idle", 100, seed=0)
    assert r["wins"] == r["games"]


@pytest.mark.parametrize("name", ["camper", "rusher", "kiter"])
def test_new_bots_never_lose_to_random(name):
    r = play_matches(name, "random", 200, seed=0)
    assert r["losses"] == 0


def test_rule_bot_still_beats_its_variants():
    # They shoot like the rule bot but move worse against it.
    for name in ["camper", "rusher", "kiter"]:
        r = play_matches("rule", name, 200, seed=0)
        assert r["wins"] > r["losses"]


def mean_fighting_distance(name: str, games: int = 64, steps: int = 400) -> float:
    """Average distance to the rule bot while the two can see each other."""
    env = ArenaEnv(games, seed=0, auto_reset=False)
    bots = [BOTS[name](env, 0, 1), RuleBot(env, 1, 2)]
    playing, seen = np.ones(games, dtype=bool), []
    for _ in range(steps):
        _, done, _ = env.step(np.stack([b.act(env) for b in bots], axis=1))
        both = env.visible & playing
        seen.append(np.linalg.norm(env.pos[both, 0] - env.pos[both, 1], axis=-1))
        playing &= ~done
    return float(np.concatenate(seen).mean())


def test_rusher_fights_closer_and_kiter_further_than_the_rule_bot():
    assert mean_fighting_distance("rusher") < mean_fighting_distance("rule") < mean_fighting_distance("kiter")


def test_camper_stays_near_its_spawn():
    games = 64
    env = ArenaEnv(games, seed=0, auto_reset=False)
    bots = [CamperBot(env, 0, 1), RuleBot(env, 1, 2)]
    home = env.pos[:, 0].copy()
    farthest, playing = np.zeros(games), np.ones(games, dtype=bool)
    for _ in range(300):
        _, done, _ = env.step(np.stack([b.act(env) for b in bots], axis=1))
        moved = np.linalg.norm(env.pos[:, 0] - home, axis=-1)
        farthest = np.maximum(farthest, np.where(playing, moved, 0.0))
        playing &= ~done
    assert np.median(farthest) < 4.0  # the rule bot walks 15 or more
