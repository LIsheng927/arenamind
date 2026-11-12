"""Checks for exporting a policy to the Unity game: the ONNX file and arenamind.json."""

import json
import sys

import numpy as np
import pytest

import tools.export_unity as export_unity
from tools.export_unity import export, level_pairs, levels, personalities
from tools.onnx_mlp import _varint, forward, mlp_model


def random_layers(sizes, seed=0):
    rng = np.random.default_rng(seed)
    return [(rng.normal(0, 0.3, (n_out, n_in)).astype(np.float32),
             rng.normal(0, 0.1, n_out).astype(np.float32)) for n_in, n_out in zip(sizes, sizes[1:])]


LADDER = {
    "checkpoint": "runs/x/model.pt", "knob": "every",
    "levels": [{"name": "L5", "level": 1.0, "elo": 1550.0}, {"name": "L4", "level": 1.4, "elo": 1300.0},
               {"name": "L1", "level": 3.9, "elo": 700.0}],
    "also": [{"level": "L5", "settings": "aggression=-1", "elo": 1580.0},
             {"level": "L1", "settings": "aggression=1", "elo": 660.0}],
}


def test_varints_are_encoded_like_protocol_buffers():
    assert _varint(1) == b"\x01"
    assert _varint(300) == b"\xac\x02"
    assert len(_varint(-1)) == 10  # a negative int64 always takes ten bytes


def test_the_onnx_file_runs_in_onnxruntime_like_the_layers():
    ort = pytest.importorskip("onnxruntime")
    layers = random_layers([51, 32, 32, 18])
    session = ort.InferenceSession(mlp_model(layers, metadata={"note": "test"}),
                                   providers=["CPUExecutionProvider"])
    (inp,), (out,) = session.get_inputs(), session.get_outputs()
    assert (inp.name, inp.shape[1], out.name, out.shape[1]) == ("obs", 51, "logits", 18)
    assert session.get_modelmeta().custom_metadata_map == {"note": "test"}
    for batch in (1, 5):  # the batch size is free
        x = np.random.default_rng(batch).normal(size=(batch, 51)).astype(np.float32)
        got = session.run(None, {"obs": x})[0]
        assert got.shape == (batch, 18)
        assert np.abs(got - forward(layers, x)).max() < 1e-5


def test_layers_that_do_not_fit_together_are_refused():
    first, second = random_layers([51, 32, 18])
    with pytest.raises(ValueError):
        mlp_model([first, (second[0][:, :20], second[1])])


def test_levels_are_easiest_first_with_a_rating_for_each_personality():
    offered = personalities(2)
    assert [p["name"] for p in offered] == ["careful", "neutral", "aggressive"]
    assert [p["sliders"] for p in offered] == [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]
    found = levels(LADDER, offered)
    assert [lv["name"] for lv in found] == ["L1", "L4", "L5"]
    assert found[0] == {"name": "L1", "every": 3.9, "elo": [0.0, 700.0, 660.0]}  # 0: not rated
    assert found[2]["elo"] == [1580.0, 1550.0, 0.0]
    assert levels(None, offered) == [{"name": "full", "every": 1.0, "elo": [0.0, 0.0, 0.0]}]
    with pytest.raises(ValueError):
        levels({**LADDER, "knob": "delay"}, offered)


CHECKS_CSV = """stronger,weaker,rating_gap,predicted,actual,noise,neighbours
L5,L4,251.4,0.8096,0.9062,0.0146,True
L5,L1,855.0,0.9928,1.0,0.0,False
L4,L1,603.6,0.97,0.9912,0.0047,True
"""


def test_the_levels_against_each_other_are_read_from_checks_csv(tmp_path):
    assert level_pairs(tmp_path / "checks.csv") == []  # a ladder built before the league checks
    (tmp_path / "checks.csv").write_text(CHECKS_CSV)
    pairs = level_pairs(tmp_path / "checks.csv")
    assert pairs == [{"stronger": "L5", "weaker": "L4", "actual": 0.9062, "noise": 0.0146},
                     {"stronger": "L4", "weaker": "L1", "actual": 0.9912, "noise": 0.0047}]


def test_export_writes_the_network_and_what_the_game_needs(tmp_path):
    layers = random_layers([51, 16, 16, 18])
    pairs = [{"stronger": "L5", "weaker": "L4", "actual": 0.9, "noise": 0.01}]
    report = export(layers, lambda x: forward(layers, x), tmp_path, "runs/x/model.pt", LADDER, checks=6,
                    pairs=pairs)
    assert report["numpy"] == 0.0
    data = json.loads((tmp_path / "arenamind.json").read_text())
    assert (data["inputs"], data["observation"], data["actionSizes"]) == (51, 49, [9, 5, 2, 2])
    assert data["sliders"] == ["aggression", "discipline"]
    assert [lv["name"] for lv in data["levels"]] == ["L1", "L4", "L5"]
    assert data["pairs"] == pairs
    checks = data["checks"]
    assert len(checks) == 6 and all(len(c["input"]) == 51 and len(c["logits"]) == 18 for c in checks)
    assert [c["input"][49] for c in checks[:3]] == [-1.0, 0.0, 1.0]  # every personality is checked
    x = np.array([c["input"] for c in checks], dtype=np.float32)
    assert np.abs(forward(layers, x) - np.array([c["logits"] for c in checks])).max() < 1e-5
    assert (tmp_path / "policy.onnx").stat().st_size == report["bytes"]


def test_a_policy_without_sliders_offers_its_own_play_only(tmp_path):
    layers = random_layers([49, 16, 18])
    export(layers, lambda x: forward(layers, x), tmp_path, checks=3)
    data = json.loads((tmp_path / "arenamind.json").read_text())
    assert data["inputs"] == 49 and data["sliders"] == []
    assert data["personalities"] == [{"name": "neutral", "settings": "", "sliders": []}]


def test_an_export_that_does_not_match_the_policy_is_refused(tmp_path):
    layers = random_layers([51, 16, 18])
    with pytest.raises(ValueError):
        export(layers, lambda x: forward(layers, x) + 0.01, tmp_path, checks=3)
    with pytest.raises(ValueError):  # not a policy for this game: 17 scores
        export(random_layers([51, 16, 17]), lambda x: x, tmp_path, checks=3)
    assert not (tmp_path / "policy.onnx").exists()


def test_the_command_line_with_a_saved_policy(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    from rl.model import ActorCritic
    from rl.ppo import PPOConfig, save_checkpoint

    path = tmp_path / "model.pt"
    save_checkpoint(path, ActorCritic(51, (9, 5, 2, 2), hidden=16), PPOConfig(hidden=16), step=0)
    ladder = tmp_path / "ladder.json"
    ladder.write_text(json.dumps({**LADDER, "checkpoint": str(path)}))
    (tmp_path / "checks.csv").write_text(CHECKS_CSV)
    monkeypatch.setattr(sys, "argv", ["export_unity", str(path), "--ladder", str(ladder),
                                      "--out", str(tmp_path / "out"), "--checks", "4"])
    export_unity.main()
    assert "wrote" in capsys.readouterr().out
    assert (tmp_path / "out" / "policy.onnx").stat().st_size > 0
    data = json.loads((tmp_path / "out" / "arenamind.json").read_text())
    assert len(data["checks"]) == 4 and [p["weaker"] for p in data["pairs"]] == ["L4", "L1"]
