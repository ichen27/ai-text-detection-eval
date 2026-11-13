"""Detection metrics. Scores are "higher means more likely AI"; label 1 is AI, 0 is human."""
from __future__ import annotations

import numpy as np


def auroc(human: np.ndarray, ai: np.ndarray) -> float:
    """Probability that a random AI text scores above a random human text (ties count half)."""
    human, ai = np.asarray(human, float), np.asarray(ai, float)
    if human.size == 0 or ai.size == 0:
        raise ValueError("both groups need at least one score")
    order = np.sort(human)
    below = np.searchsorted(order, ai, side="left")
    ties = np.searchsorted(order, ai, side="right") - below
    return float((below + 0.5 * ties).sum() / (human.size * ai.size))


def threshold_for_fpr(human: np.ndarray, fpr: float) -> float:
    """Smallest threshold t such that flagging ``score > t`` flags at most ``fpr`` of human texts."""
    human = np.sort(np.asarray(human, float))
    if human.size == 0 or not 0 <= fpr < 1:
        raise ValueError("need human scores and 0 <= fpr < 1")
    allowed = int(np.floor(fpr * human.size))
    return float(human[human.size - 1 - allowed])


def tpr_at_fpr(human: np.ndarray, ai: np.ndarray, fpr: float) -> float:
    """Share of AI texts flagged when the threshold is set so at most ``fpr`` of humans are flagged."""
    return float((np.asarray(ai, float) > threshold_for_fpr(human, fpr)).mean())


def flag_rate(scores: np.ndarray, threshold: float) -> float:
    return float((np.asarray(scores, float) > threshold).mean())


def bootstrap_ci(statistic, human: np.ndarray, ai: np.ndarray, n: int = 2000, seed: int = 0,
                 level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval, resampling each group independently."""
    rng = np.random.default_rng(seed)
    human, ai = np.asarray(human, float), np.asarray(ai, float)
    values = [statistic(rng.choice(human, human.size), rng.choice(ai, ai.size)) for _ in range(n)]
    low, high = np.quantile(values, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(low), float(high)


def calibration(scores: np.ndarray, labels: np.ndarray, bins: int = 5) -> list[dict]:
    """Per equal-width score bin: count, mean score and the actual share of AI texts."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, float)
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(scores, edges[1:-1]), 0, bins - 1)
    table = []
    for b in range(bins):
        mask = index == b
        table.append({"low": float(edges[b]), "high": float(edges[b + 1]), "n": int(mask.sum()),
                      "mean_score": float(scores[mask].mean()) if mask.any() else None,
                      "share_ai": float(labels[mask].mean()) if mask.any() else None})
    return table


def expected_calibration_error(scores: np.ndarray, labels: np.ndarray, bins: int = 5) -> float:
    total = len(scores)
    return float(sum(row["n"] / total * abs(row["mean_score"] - row["share_ai"])
                     for row in calibration(scores, labels, bins) if row["n"]))


def held_out_rates(calib_human: np.ndarray, test_human: np.ndarray, test_ai: np.ndarray, fpr: float) -> dict:
    """Set the threshold on one group of human texts and measure on texts it has not seen.

    Returns the threshold, the share of test AI texts flagged, and the share of test human texts
    flagged (which can differ from the target ``fpr``; that gap is the point of holding texts out)."""
    threshold = threshold_for_fpr(calib_human, fpr)
    return {"threshold": threshold, "tpr": flag_rate(test_ai, threshold), "fpr": flag_rate(test_human, threshold)}


def fit_logistic(features: np.ndarray, labels: np.ndarray, l2: float = 1.0, iterations: int = 100) -> dict:
    """L2-regularised logistic regression by Newton's method on standardised features."""
    features, labels = np.asarray(features, float), np.asarray(labels, float)
    if features.ndim != 2 or features.shape[0] != labels.shape[0] or len(set(labels.tolist())) < 2:
        raise ValueError("need a 2-D feature matrix and labels containing both classes")
    mean, scale = features.mean(axis=0), features.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    design = np.hstack([np.ones((features.shape[0], 1)), (features - mean) / scale])
    penalty = l2 * np.eye(design.shape[1])
    penalty[0, 0] = 0.0                      # the intercept is not penalised
    weights = np.zeros(design.shape[1])
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-np.clip(design @ weights, -30, 30)))
        gradient = design.T @ (p - labels) + penalty @ weights
        hessian = (design * (p * (1 - p))[:, None]).T @ design + penalty
        step = np.linalg.solve(hessian, gradient)
        weights -= step
        if np.abs(step).max() < 1e-8:
            break
    return {"mean": mean, "scale": scale, "weights": weights}


def predict_logistic(model: dict, features: np.ndarray) -> np.ndarray:
    """Log-odds of the positive class for each row."""
    features = (np.asarray(features, float) - model["mean"]) / model["scale"]
    return model["weights"][0] + features @ model["weights"][1:]


def wilson_interval(flagged: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion; (0, 1) when there is no data."""
    if total == 0:
        return 0.0, 1.0
    p = flagged / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))
