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
