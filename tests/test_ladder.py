"""Checks for building a difficulty ladder, with the ratings faked."""

import json
import math
import sys

import numpy as np
import pytest

import tools.difficulty_ladder as ladder
from rl.rate import Style
from tools.difficulty_ladder import expected_score, find_setting, pair_checks


def curve(level):
    """A made-up rating that falls quickly at first and then levels off, like every's."""
    return 480 + 1020 * math.exp(-1.2 * (level - 1))


def test_expected_score():
    assert expected_score(1000, 1000) == 0.5
    assert abs(expected_score(1200, 1000) - 0.760) < 0.001
    assert abs(expected_score(1000, 1200) + expected_score(1200, 1000) - 1) < 1e-12


def test_each_target_is_found_within_tolerance_and_tries_are_shared():
    tried = {1.0: curve(1.0), 5.0: curve(5.0)}
    calls = []

    def rate(level):
        calls.append(level)
        return curve(level)

    for target in (1300, 1100, 900, 700):
        level = find_setting(target, rate, tried, step=0.05, tolerance=25)
        assert abs(curve(level) - target) <= 25
        assert level in tried and round(level / 0.05, 6) % 1 == 0  # a whole number of steps
    assert len(calls) == len(tried) - 2 and len(calls) <= 14


def test_the_search_stops_when_the_target_falls_between_two_neighbouring_steps():
    def rate(level):
        return 1500.0 if level < 2.0 else 1000.0  # a cliff: nothing in between

    tried = {1.0: 1500.0, 5.0: 1000.0}
    level = find_setting(1260, rate, tried, step=0.05, tolerance=25, max_tries=20)
    assert 1.95 in tried and 2.0 in tried  # it closed in on the cliff to a single step
    assert tried[level] == 1500.0  # and returned the closer side
    assert len(tried) < 12


def test_a_target_outside_the_tried_ratings_is_refused():
    with pytest.raises(ValueError):
        find_setting(1600, curve, {1.0: 1500.0, 5.0: 500.0}, step=0.05)


def test_pair_checks():
    score = np.array([[0.0, 300.0], [100.0, 0.0]])
    count = np.array([[0.0, 400.0], [400.0, 0.0]])
    (row,) = pair_checks(["L2", "L1"], [1200.0, 1000.0], score, count)
    assert row["stronger"] == "L2" and row["neighbours"]
    assert row["actual"] == 0.75 and abs(row["predicted"] - 0.76) < 0.001
    assert abs(row["noise"] - math.sqrt(0.75 * 0.25 / 400)) < 1e-4


def test_the_whole_run_with_ratings_faked(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")

    def level_of(spec):
        return float(spec.split("@every=")[1]) if "@" in spec else 1.0

    def fake_rate(spec, ratings, games, seed):
        style = Style(seconds=100.0, shots=200, hits=20, blind_shots=90, move_changes=500,
                      turn_reversals=round(370 / level_of(spec)), reactions=[0.2 + 0.03 * level_of(spec)])
        return curve(level_of(spec)), {o: 0.5 for o in ratings}, style

    def fake_league(players, games, seed):
        r = np.array([curve(level_of(p)) for p in players])
        count = 2.0 * games * (1 - np.eye(len(players)))
        return count * expected_score(r[:, None], r[None, :]), count

    monkeypatch.setattr(ladder, "rate_player", fake_rate)
    monkeypatch.setattr(ladder, "run_league", fake_league)
    monkeypatch.setattr(ladder, "panel_ratings", lambda *a, **k: {"random": 477.0, "rule": 1000.0})
    monkeypatch.setattr(sys, "argv", ["difficulty_ladder", "runs/x/model.pt", "--out", str(tmp_path)])
    ladder.main()

    saved = json.loads((tmp_path / "ladder.json").read_text())
    levels = saved["levels"]
    assert [lv["name"] for lv in levels] == ["L5", "L4", "L3", "L2", "L1"]
    assert levels[0]["setting"] == "full" and levels[0]["spec"] == "ppo:runs/x/model.pt"
    assert levels[1]["spec"].startswith("ppo:runs/x/model.pt@every=")
    for lv in levels[1:]:
        assert abs(lv["elo"] - lv["target"]) <= 25
    for name in ("tries.csv", "checks.csv", "ladder.png"):
        assert (tmp_path / name).stat().st_size > 0
