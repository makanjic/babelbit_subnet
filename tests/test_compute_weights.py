import pytest

from babelbit.cli.validate import compute_weights


def test_compute_weights_with_trailing_scores():
    winner_uid = 10
    trailing = {2: 3.0, 5: 1.0}  # total trailing = 4.0

    weights, uids = compute_weights(winner_uid, trailing)

    assert uids == [10, 2, 5]
    assert weights == pytest.approx([0.95, 0.0375, 0.0125])
    assert pytest.approx(sum(weights), rel=1e-6) == 1.0


def test_compute_weights_no_trailing_scores():
    weights, uids = compute_weights(7, {})

    assert uids == [7]
    assert weights == [1.0]


def test_compute_weights_zero_or_negative_trailing():
    # Negative/None/zero trailing scores are clamped to zero and ignored in distribution
    trailing = {3: -1.0, 4: 0.0, 8: None}

    weights, uids = compute_weights(1, trailing)

    assert uids == [1]
    assert weights == [1.0]


def test_compute_weights_close_to_winner():
    winner_uid = 20
    trailing = {6: 94.0, 9: 5.0}  # total trailing = 99.0

    weights, uids = compute_weights(winner_uid, trailing)

    assert uids == [20, 6, 9]
    # Winner gets 95%; trailing miners split the remaining 5% in proportion to their scores
    assert pytest.approx(weights[0]) == 0.95
    trailing_weights = weights[1:]
    assert pytest.approx(sum(trailing_weights)) == 0.05
    # Ratio of trailing weights should match ratio of trailing scores
    assert pytest.approx(trailing_weights[0] / trailing_weights[1]) == pytest.approx(trailing[6] / trailing[9])
