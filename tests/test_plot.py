import csv

import pytest

pytest.importorskip("matplotlib")

from tools.plot_training import parse_phase, plot, read_log, smooth  # noqa: E402

COLUMNS = ["step", "win_rate", "hits_per_game", "hit_by_per_game", "mean_game_steps"]


def test_smooth_is_a_trailing_moving_average():
    import numpy as np
    assert smooth(np.array([0.0, 3.0, 6.0, 9.0]), 3).tolist() == [0.0, 1.0, 3.0, 6.0]


def test_parse_phase():
    assert parse_phase("1.3:learn to hit") == (1.3, "learn to hit")


def test_plot_writes_an_image(tmp_path):
    path = tmp_path / "log.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for i in range(1, 30):
            w.writerow([i * 32768, min(1, i / 20), i / 6, 5 - i / 6, 600 - i * 10])
    out = tmp_path / "plots" / "training.png"
    plot(read_log(path), out, "test run", [(0.0, "early"), (0.5, "late")])
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_parse_series():
    from pathlib import Path

    from tools.plot_logs import parse_series
    assert parse_series("main=runs/a/log.csv:win_rate") == ("main", Path("runs/a/log.csv"), "win_rate")
    assert parse_series(r"C:\runs\a\log.csv:eval_rule")[1:] == (Path(r"C:\runs\a\log.csv"), "eval_rule")
    assert parse_series("runs/a/log.csv:win_rate")[0] == "a win_rate"


def test_spread_keeps_labels_apart_and_below_the_top():
    from tools.plot_logs import spread
    placed = spread([1.0, 1.0, 0.2], gap=0.1, top=1.0)
    assert placed[2] == 0.2 and max(placed) <= 1.0
    assert abs(placed[0] - placed[1]) >= 0.1 - 1e-9


def test_plot_logs_writes_an_image(tmp_path):
    from tools.plot_logs import curve, plot
    path = tmp_path / "log.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "win_rate", "eval_rule"])
        for i in range(1, 30):
            w.writerow([i * 32768, i / 30, "nan" if i < 10 else round(i / 30, 1)])
    log = read_log(path)
    x, y, steps = curve(log, "eval_rule", window=5)
    assert len(x) == 20 and steps  # nan rows left out, evaluations drawn as steps
    assert not curve(log, "win_rate", window=5)[2]
    out = tmp_path / "compare.png"
    plot([("win", log, "win_rate"), ("eval", log, "eval_rule")], out, "test", percent=True,
         marks=[(0.5, "even")])
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_difficulty_charts_are_redrawn_from_a_results_file(tmp_path):
    from rl.rate import write_csv
    from tools.difficulty_sweep import plot, read_rows

    style = {"accuracy": 0.09, "reaction_s": 0.22, "move_changes_per_s": 5.7,
             "turn_reversals_per_s": 3.7, "blind_shot_share": 0.48}
    rows = [{"knob": "full", "level": 0, "spec": "ppo:m.pt", "elo": 1500.0, **style},
            {"knob": "every", "level": 2, "spec": "ppo:m.pt@every=2", "elo": 1180.0, **style},
            {"knob": "every", "level": 3, "spec": "ppo:m.pt@every=3", "elo": 790.0, **style}]
    write_csv(rows, tmp_path / "results.csv")
    back = read_rows(tmp_path / "results.csv")
    assert back[1]["level"] == 2.0 and back[1]["knob"] == "every" and back[2]["elo"] == 790.0
    plot(back, {"rule": 1000.0, "rusher": 1007.0, "random": 477.0}, ["every"], tmp_path)
    assert (tmp_path / "strength.png").stat().st_size > 0 and (tmp_path / "style.png").stat().st_size > 0
