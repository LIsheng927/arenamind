"""Checks for the games recorded for the Unity port's tests (tools/export_traces.py)."""

import json
import sys

import numpy as np

import tools.export_traces as export_traces
from arena.config import GameConfig
from arena.env import ArenaEnv
from arena.obs import observe
from tools.export_traces import config_numbers, controls, gate_pattern, record_game

CFG = GameConfig()


def replay(game):
    """Start a fresh game where the recorded one started and repeat its actions, like the C# tests."""
    env = ArenaEnv(1, seed=12345, auto_reset=False)
    env.pos[0] = np.reshape(game["startPos"], (2, 2))
    env.angle[0] = game["startAngle"]
    env._update_sight(np.arange(1), tick=False)  # what reset() does after placing the players
    actions = np.reshape(game["actions"], (-1, 1, 2, 4))
    positions, observations = [], []
    for a in actions:
        env.step(a)
        positions.append(env.pos[0].ravel().copy())
        observations.append(observe(env)[0].ravel())
    return np.array(positions), np.array(observations)


def test_a_recorded_game_can_be_replayed_from_its_start_and_actions():
    game = record_game("rule", "rusher", seed=0)
    n = game["steps"]
    sizes = {"actions": 8, "pos": 4, "angle": 2, "hp": 2, "timers": 6, "bullets": 6, "bulletPos": 12,
             "visible": 1, "unseen": 2, "lastSeen": 4, "done": 1, "outcome": 1, "obs": 2 * CFG.obs.obs_dim}
    for key, size in sizes.items():
        assert len(game[key]) == n * size, key
    assert game["done"][-1] == 1 and sum(game["done"]) == 1  # recorded until the game ended
    assert min(game["hp"][-2:]) == 0.0  # it ended with a player at 0 HP

    positions, observations = replay(game)
    assert np.array_equal(positions.ravel(), np.array(game["pos"]))
    assert np.allclose(observations.ravel(), game["obs"], atol=1e-7, rtol=0)


def test_values_are_written_so_that_they_read_back_the_same():
    game = record_game("random", "rule", seed=1)
    env = ArenaEnv(1, seed=1, auto_reset=False)
    start = observe(env)[0].ravel()
    assert np.array_equal(np.array(game["startObs"], dtype=np.float32), start)  # float32 exactly
    assert game["startPos"] == [float(v) for v in env.pos[0].ravel()]  # doubles exactly


def test_a_policy_deciding_every_n_steps():
    assert gate_pattern(1.0, CFG.dt) == [1] * 40
    pattern = gate_pattern(2.5, CFG.dt)
    assert pattern[:6] == [1, 0, 0, 1, 0, 1]  # gaps of 3 and 2 steps: 2.5 on average
    assert pattern[25] == 1  # a new game always starts with a decision


def test_controls_and_config():
    c = controls(CFG)
    assert len(c["turnDiffs"]) == len(c["turns"]) and len(c["moveAngles"]) == len(c["moves"])
    assert set(c["moves"]) <= set(range(1, 9)) and set(c["turns"]) <= set(range(5))
    assert c["turnsForDegrees"] == [0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 4]  # ties fall either way
    eighths = dict(zip(c["moveEighths"], c["movesForEighths"]))
    assert (eighths[1], eighths[-1], eighths[3], eighths[-3]) == (1, 1, 3, 7)  # halfway: rounded to even
    assert all(-np.pi <= a <= np.pi for a in c["wrapOut"])
    numbers = config_numbers(CFG)
    assert numbers["obsDim"] == 49 and numbers["maxSteps"] == 600 and len(numbers["obstacles"]) == 20


def test_the_command_line_writes_every_game(tmp_path, monkeypatch, capsys):
    games = (("rule", "rusher", 0), ("idle", "idle", 3))
    monkeypatch.setattr(export_traces, "GAMES", games)
    monkeypatch.setattr(sys, "argv", ["export_traces", "--out", str(tmp_path / "traces.json")])
    export_traces.main()
    data = json.loads((tmp_path / "traces.json").read_text())
    assert [g["name"] for g in data["games"]] == ["rule vs rusher", "idle vs idle"]
    assert data["games"][1]["steps"] == CFG.max_steps  # nobody shoots: the time limit ends it
    assert data["games"][1]["outcome"][-1] == 0
    assert "wrote" in capsys.readouterr().out
