"""Checks for the personality settings that do not need PyTorch."""

import math

import numpy as np
import pytest

from arena.config import GameConfig
from rl.difficulty import OPPONENT_POSITION, Difficulty
from rl.personality import OPPONENT_VISIBLE, Personality, parse_settings, settings_label
from tools.personality_sweep import settings, verdicts

CFG = GameConfig()


def obs_with_opponent(forward, right, visible=True, n=1):
    obs = np.zeros((n, CFG.obs.obs_dim), dtype=np.float32)
    obs[:, OPPONENT_POSITION[0]], obs[:, OPPONENT_POSITION[1]] = forward, right
    obs[:, OPPONENT_VISIBLE] = 1.0 if visible else 0.0
    return obs


def test_settings_are_split_between_difficulty_and_personality():
    knobs, personality = parse_settings("every=2.1,hold=4")
    assert knobs == Difficulty(every=2.1) and personality == Personality(hold=4.0)
    assert settings_label(knobs, personality) == "every=2.1,hold=4"
    assert parse_settings("approach=-1.5") == (Difficulty(), Personality(approach=-1.5))
    assert settings_label(Difficulty(), Personality()) == "full"
    assert not Personality() and Personality(dash=1)


@pytest.mark.parametrize("text", ["speed=2", "hold", "every=2,courage=1"])
def test_unknown_settings_are_refused(text):
    with pytest.raises(ValueError):
        parse_settings(text)


def test_approach_favours_moves_towards_the_opponent():
    move, turn, shoot, dash = Personality(approach=2).biases(obs_with_opponent(0.5, 0.0))
    assert move.shape == (1, 9) and not turn.any() and not shoot.any() and not dash.any()
    d = 2 / math.sqrt(2)
    # stay, forward, forward-right, right, back-right, back, back-left, left, forward-left
    assert np.allclose(move[0], [0, 2, d, 0, -d, -2, -d, 0, d])
    away, *_ = Personality(approach=-2).biases(obs_with_opponent(0.0, -0.3))  # opponent to the left
    assert away[0, 7] == pytest.approx(-2) and away[0, 3] == pytest.approx(2)


def test_no_direction_when_standing_where_the_opponent_was_seen():
    move, *_ = Personality(approach=3).biases(obs_with_opponent(0.0, 0.0))
    assert not move.any()


def test_hold_only_applies_while_the_opponent_is_out_of_sight():
    _, _, seen, _ = Personality(hold=4).biases(obs_with_opponent(0.5, 0.0, visible=True))
    _, _, unseen, _ = Personality(hold=4).biases(obs_with_opponent(0.5, 0.0, visible=False))
    assert not seen.any()
    assert unseen[0].tolist() == [0.0, -4.0]  # only "shoot" is pushed down


def test_dash_shifts_the_dash_choice():
    *_, dash = Personality(dash=-1.5).biases(obs_with_opponent(0.5, 0.0, n=3))
    assert dash.tolist() == [[0.0, -1.5]] * 3


def test_sweep_settings_and_verdicts():
    todo = settings("m.pt", ["hold"])
    assert todo[0] == ("full", 0, "ppo:m.pt") and todo[1] == ("hold", 1, "ppo:m.pt@hold=1")

    def row(trait, level, elo, **style):
        base = {"distance": 10.0, "blind_shot_share": 0.5, "dashes_per_s": 0.2}
        return {"trait": trait, "level": level, "elo": elo, **{**base, **style}}
    rows = [row("full", 0, 1500),
            row("approach", 3, 1450, distance=8.0), row("approach", -3, 1420, distance=12.0),
            row("hold", 4, 1480, blind_shot_share=0.3), row("hold", 8, 1300, blind_shot_share=0.1),
            row("dash", 3, 1490, dashes_per_s=0.25)]
    result = verdicts(rows)
    assert result["approach"][0]  # 20% closer and 20% farther, both cheap
    assert not result["hold"][0]  # halved only at a level that costs 200 points
    assert not result["dash"][0]
