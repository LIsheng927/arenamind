import numpy as np

from rl.league import fit_elo, run_league


def expected(diff):
    return 1 / (1 + 10 ** (-diff / 400))


def test_fit_recovers_known_ratings():
    true = np.array([1000.0, 1200.0, 1400.0])
    games = 100_000
    count = np.full((3, 3), float(games)) * (1 - np.eye(3))
    score = count * expected(true[:, None] - true[None, :])
    rating = fit_elo(score, count, anchor=0, anchor_rating=1000.0)
    assert np.allclose(rating, true, atol=2.0)


def test_every_player_scores_what_its_rating_predicts():
    # The defining property of the fit, on uneven results.
    score = np.array([[0.0, 70.0, 95.0], [30.0, 0.0, 60.0], [5.0, 40.0, 0.0]])
    count = 100.0 * (1 - np.eye(3))
    rating = fit_elo(score, count)
    prior = 1 - np.eye(3)  # the one extra draw per pair
    exp = expected(rating[:, None] - rating[None, :]) * (count + prior)
    assert np.allclose((exp * prior).sum(axis=1), (score + 0.5 * prior).sum(axis=1), atol=1e-6)


def test_a_perfect_record_still_gives_a_finite_rating():
    score = np.array([[0.0, 50.0], [0.0, 0.0]])
    count = 50.0 * (1 - np.eye(2))
    rating = fit_elo(score, count)
    assert np.isfinite(rating).all() and rating[0] - rating[1] > 400


def test_league_ranks_the_bots_in_the_expected_order():
    players = ["random", "camper", "rule"]
    score, count = run_league(players, games=40, seed=0)
    assert np.allclose(score + score.T, count)  # every game gives out exactly one point
    rating = fit_elo(score, count, anchor=2)
    assert rating[2] > rating[1] > rating[0]
