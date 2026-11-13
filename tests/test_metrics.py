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


def test_held_out_rates_use_the_calibration_threshold():
    calib = np.arange(100) / 100          # 5% FPR threshold is the 6th largest value, 0.94
    rates = metrics.held_out_rates(calib, np.array([0.1, 0.5, 0.95, 0.99]), np.array([0.9, 0.96, 0.97, 0.2]), 0.05)
    assert rates["threshold"] == pytest.approx(0.94)
    assert rates["fpr"] == pytest.approx(0.5)
    assert rates["tpr"] == pytest.approx(0.5)


def test_logistic_regression_separates_and_ignores_noise():
    rng = np.random.default_rng(0)
    labels = np.repeat([0, 1], 200)
    signal = labels * 2.0 + rng.normal(size=400)
    noise = rng.normal(size=400)
    model = metrics.fit_logistic(np.column_stack([signal, noise]), labels)
    scores = metrics.predict_logistic(model, np.column_stack([signal, noise]))
    assert metrics.auroc(scores[labels == 0], scores[labels == 1]) > 0.9
    assert abs(model["weights"][1]) > 5 * abs(model["weights"][2])


def test_logistic_regression_rejects_one_class():
    with pytest.raises(ValueError):
        metrics.fit_logistic(np.zeros((4, 1)), np.zeros(4))


def test_wilson_interval_contains_the_estimate_and_narrows():
    low, high = metrics.wilson_interval(5, 100)
    assert low < 0.05 < high
    wide = metrics.wilson_interval(1, 20)
    assert (high - low) < (wide[1] - wide[0])
    assert metrics.wilson_interval(0, 0) == (0.0, 1.0)
