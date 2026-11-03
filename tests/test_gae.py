import numpy as np
import pytest

from rl.gae import compute_gae


def col(values):
    """A single game as a (T, 1) column."""
    return np.array(values, dtype=np.float64)[:, None]


def test_lambda_one_is_return_minus_value():
    # With lam = 1 the advantage is the discounted return minus V.
    r = col([0.0, 0.0, 0.2, 0.0, 1.2])
    v = col([0.1, 0.2, 0.3, 0.4, 0.5])
    d = col([0, 0, 0, 0, 1])
    adv, ret = compute_gae(r, v, d, np.zeros(1), gamma=0.9, lam=1.0)
    # Returns worked out by hand: 1.2, 1.08, 1.172, 1.0548, 0.94932.
    assert ret[:, 0] == pytest.approx([0.94932, 1.0548, 1.172, 1.08, 1.2])
    assert adv[:, 0] == pytest.approx(ret[:, 0] - v[:, 0])


def test_one_step_surprise_examples():
    # lam = 0: the advantage is just delta = r + gamma * V(next) - V(now).
    r = col([0.0, -0.2, 0.0])
    v = col([0.30, 0.30, 0.30])
    d = col([0, 0, 0])
    adv, _ = compute_gae(r, v, d, last_values=np.array([0.40]), gamma=0.99, lam=0.0)
    assert adv[0, 0] == pytest.approx(0.99 * 0.30 - 0.30)  # quiet step: about 0
    assert adv[1, 0] == pytest.approx(-0.2 + 0.99 * 0.30 - 0.30)  # got hit
    assert adv[2, 0] == pytest.approx(0.99 * 0.40 - 0.30)  # situation improved


def test_credit_travels_back_to_the_shot():
    # Deltas 0, -0.01, 0, 0.2 with gamma * lam = 0.94 give 0.157 at the shot.
    gamma, lam = 0.99, 0.95
    deltas = [0.0, -0.01, 0.0, 0.2]
    # Build rewards so that delta(t) comes out exactly as listed, with V = 0.
    r = col(deltas)
    v = col([0.0] * 4)
    d = col([0, 0, 0, 0])
    adv, _ = compute_gae(r, v, d, np.zeros(1), gamma, lam)
    k = gamma * lam
    expected = [0.2 * k ** 3 - 0.01 * k, 0.2 * k ** 2 - 0.01, 0.2 * k, 0.2]
    assert adv[:, 0] == pytest.approx(expected)


def test_nothing_leaks_across_the_end_of_a_game():
    # Game ends at step 1; step 2 belongs to the next game and is negative.
    r = col([0.0, 1.0, -0.2])
    v = col([0.0, 0.0, 0.0])
    d = col([0, 1, 0])
    adv, ret = compute_gae(r, v, d, np.zeros(1), gamma=0.99, lam=0.95)
    assert ret[1, 0] == pytest.approx(1.0)  # not touched by the next game's -0.2
    assert ret[0, 0] == pytest.approx(0.99 * 0.95 * 1.0)


def test_unfinished_games_use_the_critic_for_what_comes_next():
    r = col([0.0])
    v = col([0.5])
    d = col([0])
    adv, ret = compute_gae(r, v, d, last_values=np.array([0.8]), gamma=0.99, lam=0.95)
    assert adv[0, 0] == pytest.approx(0.99 * 0.8 - 0.5)


def test_many_games_at_once_match_one_at_a_time():
    rng = np.random.default_rng(0)
    T, N = 50, 8
    r = rng.normal(size=(T, N)) * 0.2
    v = rng.normal(size=(T, N))
    d = rng.random((T, N)) < 0.05
    last = rng.normal(size=N)
    adv, ret = compute_gae(r, v, d, last, 0.99, 0.95)
    for n in range(N):
        a1, r1 = compute_gae(r[:, n:n + 1], v[:, n:n + 1], d[:, n:n + 1], last[n:n + 1], 0.99, 0.95)
        assert np.allclose(adv[:, n], a1[:, 0]) and np.allclose(ret[:, n], r1[:, 0])
