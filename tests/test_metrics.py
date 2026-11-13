import numpy as np
import pytest

from detecteval import metrics


def test_auroc_perfect_random_and_ties():
    assert metrics.auroc([0.1, 0.2], [0.8, 0.9]) == 1.0
    assert metrics.auroc([0.8, 0.9], [0.1, 0.2]) == 0.0
    assert metrics.auroc([0.5, 0.5], [0.5, 0.5]) == 0.5
    assert metrics.auroc([0.1, 0.5], [0.5, 0.9]) == pytest.approx(0.875)


def test_auroc_matches_pairwise_definition():
    rng = np.random.default_rng(1)
    human, ai = rng.random(40).round(1), rng.random(30).round(1)
    pairwise = np.mean([(a > h) + 0.5 * (a == h) for a in ai for h in human])
    assert metrics.auroc(human, ai) == pytest.approx(pairwise)


def test_threshold_never_exceeds_requested_fpr():
    human = np.arange(100) / 100
    for fpr in (0.0, 0.01, 0.05, 0.2):
        threshold = metrics.threshold_for_fpr(human, fpr)
        assert metrics.flag_rate(human, threshold) <= fpr


def test_threshold_with_tied_human_scores_stays_conservative():
    human = np.array([0.9] * 10 + [0.1] * 90)
    assert metrics.flag_rate(human, metrics.threshold_for_fpr(human, 0.05)) == 0.0


def test_tpr_at_fpr():
    human = np.arange(100) / 100
    ai = np.array([0.5, 0.97, 0.995, 1.0])
    assert metrics.tpr_at_fpr(human, ai, 0.0) == 0.5
    assert metrics.tpr_at_fpr(human, ai, 0.05) == 0.75


def test_calibration_and_ece():
    scores = np.array([0.1, 0.1, 0.9, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert metrics.expected_calibration_error(scores, labels) == pytest.approx(0.1)
    table = metrics.calibration(scores, labels, bins=5)
    assert [row["n"] for row in table] == [2, 0, 0, 0, 2]
    assert table[0]["share_ai"] == 0.0 and table[4]["share_ai"] == 1.0


def test_bootstrap_interval_brackets_point_estimate():
    rng = np.random.default_rng(2)
    human, ai = rng.normal(0, 1, 200), rng.normal(1, 1, 200)
    low, high = metrics.bootstrap_ci(metrics.auroc, human, ai, n=300)
    assert low < metrics.auroc(human, ai) < high


def test_empty_groups_rejected():
    with pytest.raises(ValueError):
        metrics.auroc([], [0.5])
