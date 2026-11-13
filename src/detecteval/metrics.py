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
