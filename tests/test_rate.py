"""Checks for rating players against a fixed panel."""

import json

import numpy as np

import rl.rate as rate
from rl.rate import Style, fit_against, panel_ratings, play_measured, short


def test_fit_against_recovers_a_known_rating():
    ratings = {"a": 1000.0, "b": 1200.0, "c": 1400.0}
    true = 1250.0
    count = {k: 4000.0 for k in ratings}
    score = {k: count[k] / (1 + 10 ** ((r - true) / 400)) for k, r in ratings.items()}
    assert abs(fit_against(score, count, ratings) - true) < 3


def test_a_perfect_record_still_gives_a_finite_rating():
    ratings = {"a": 1000.0}
    assert np.isfinite(fit_against({"a": 100.0}, {"a": 100.0}, ratings))


def test_style_of_a_bot_that_never_moves_or_shoots():
    style = Style()
    w, l, d = play_measured("idle", "rule", 20, seed=0, side=0, style=style)
    assert w + l + d == 20 and w == 0
    assert style.shots == 0 and style.move_changes == 0 and style.turn_reversals == 0
    assert style.seconds > 0


def test_style_of_the_rule_bot_is_sensible():
    style = Style()
    w, l, d = play_measured("rule", "idle", 30, seed=1, side=1, style=style)
    assert w == 30  # the idle bot never shoots back
    s = style.summary()
    assert 0 < s["accuracy"] <= 1
    assert style.hits >= 5 * 30  # five hits win a game
    assert 0 < s["reaction_s"] < 5
    assert 0 <= s["blind_shot_share"] <= 1


def test_panel_ratings_are_saved_and_reused(tmp_path, monkeypatch):
    path = tmp_path / "panel.json"
    first = panel_ratings(["random", "rule"], games=10, seed=0, path=path)
    assert first["rule"] == 1000.0 and first["random"] < 1000.0
    assert json.loads(path.read_text())["players"] == ["random", "rule"]

    def no_league(*args, **kwargs):
        raise AssertionError("the saved ratings should have been reused")
    monkeypatch.setattr(rate, "run_league", no_league)
    assert panel_ratings(["random", "rule"], games=10, seed=0, path=path) == first


def test_short_names():
    assert short("rule") == "rule"
    assert short("ppo:runs/selfplay3/model.pt") == "selfplay3"
    assert short("ppo:runs/selfplay3/model.pt@delay=3") == "selfplay3@delay=3"
