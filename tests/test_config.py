import dataclasses
import math

import pytest

from arena.config import (
    MOVE_DIRECTIONS,
    AgentConfig,
    ArenaConfig,
    DashConfig,
    GameConfig,
    WeaponConfig,
    seconds_to_steps,
)


def test_default_config_derived_values():
    cfg = GameConfig()
    assert cfg.max_steps == 600
    assert cfg.shoot_cooldown_steps == 4
    assert cfg.bullet_lifetime_steps == 10
    assert cfg.dash_steps == 3
    assert cfg.dash_cooldown_steps == 30
    assert cfg.move_per_step == pytest.approx(0.4)
    assert cfg.bullet_per_step == pytest.approx(1.5)
    assert cfg.obs.obs_dim == 49
    assert cfg.action_sizes == (9, 5, 2, 2)
    assert cfg.dash_substeps == 3


def test_move_directions_have_unit_length():
    assert MOVE_DIRECTIONS[0] == (0.0, 0.0)
    for fwd, right in MOVE_DIRECTIONS[1:]:
        assert math.hypot(fwd, right) == pytest.approx(1.0)


def test_seconds_to_steps_survives_float_error():
    # 0.3 / 0.1 is 2.9999999999999996 in floating point; int() would give 2.
    assert seconds_to_steps(0.3, 0.1, "x") == 3


def test_rejects_duration_that_is_not_whole_steps():
    with pytest.raises(ValueError, match="whole number"):
        GameConfig(weapon=WeaponConfig(cooldown=0.45))


def test_rejects_asymmetric_map():
    lopsided = ArenaConfig(obstacles=((9.0, 9.0, 11.0, 11.0), (4.0, 11.0, 8.0, 12.0)))
    with pytest.raises(ValueError, match="mirror"):
        GameConfig(arena=lopsided)


def test_rejects_spawn_inside_obstacle():
    blocked = ArenaConfig(
        obstacles=((2.0, 16.0, 4.0, 18.0), (16.0, 2.0, 18.0, 4.0)),  # covers both spawns
    )
    with pytest.raises(ValueError, match="too close to obstacle"):
        GameConfig(arena=blocked)


def test_rejects_too_few_bullet_slots():
    with pytest.raises(ValueError, match="max_bullets"):
        GameConfig(weapon=WeaponConfig(max_bullets=2))


def test_config_is_read_only():
    cfg = GameConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.dt = 0.05


def test_rejects_moves_longer_than_radius():
    with pytest.raises(ValueError, match="radius"):
        GameConfig(agent=AgentConfig(speed=6.0))  # 0.6 per step > radius 0.5


def test_rejects_fractional_dash_multiplier():
    with pytest.raises(ValueError, match="speed_multiplier"):
        GameConfig(dash=DashConfig(speed_multiplier=2.5))
