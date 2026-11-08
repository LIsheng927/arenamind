"""Checks for the personality sliders' rewards, which do not need PyTorch."""

import numpy as np
import pytest

from rl.style import (AGGRESSION_HIT, AGGRESSION_TIME, BLIND_SHOT, CAUTION_HIT, SLIDERS,
                      sample_styles, style_reward, with_style)


def test_sliders_are_drawn_over_their_ranges():
    styles = sample_styles(np.random.default_rng(0), 20000)
    assert styles.shape == (20000, len(SLIDERS))
    aggression, discipline = styles[:, 0], styles[:, 1]
    assert aggression.min() >= -1 and aggression.max() <= 1 and abs(aggression.mean()) < 0.03
    assert discipline.min() >= 0 and discipline.max() <= 1 and abs(discipline.mean() - 0.5) < 0.03


def test_neutral_sliders_add_nothing():
    zero = np.zeros((3, 2))
    assert not style_reward(zero, np.array([1.0, 0, 2]), np.array([0, 1.0, 1]), np.array([1, 0, 1])).any()


@pytest.mark.parametrize("style, dealt, taken, blind, expected", [
    ((1.0, 0.0), 1, 0, 0, AGGRESSION_HIT - AGGRESSION_TIME),  # landed a hit
    ((1.0, 0.0), 0, 1, 0, -AGGRESSION_TIME),  # an aggressive player does not mind being hit
    ((0.5, 0.0), 0, 0, 0, -0.5 * AGGRESSION_TIME),  # time costs it a little every step
    ((-1.0, 0.0), 1, 1, 0, -CAUTION_HIT),  # a careful player minds being hit, nothing else
    ((0.0, 1.0), 0, 0, 1, -BLIND_SHOT),  # a blind shot
    ((-0.5, 0.5), 0, 2, 1, -0.5 * CAUTION_HIT * 2 - 0.5 * BLIND_SHOT),
])
def test_rewards_follow_the_sliders(style, dealt, taken, blind, expected):
    got = style_reward(np.array([style]), np.array([dealt], float), np.array([taken], float),
                       np.array([blind], float))
    assert got[0] == pytest.approx(expected)


def test_sliders_are_appended_to_the_observation():
    obs = np.ones((4, 49), dtype=np.float32)
    styles = np.array([[0.5, 1.0]] * 4)
    both = with_style(obs, styles)
    assert both.shape == (4, 51) and both.dtype == np.float32 and both[0, -2:].tolist() == [0.5, 1.0]
    assert np.array_equal(with_style(obs, np.zeros((4, 0))), obs)  # without sliders: unchanged
