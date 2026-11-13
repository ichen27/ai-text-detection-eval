import numpy as np

from detecteval.binoculars import statistics
from detecteval.build_long_set import split_of
from detecteval.questions import score_of


def test_split_is_stable_and_roughly_even():
    halves = [split_of(f"doc-{n}") for n in range(2000)]
    assert split_of("doc-7") == split_of("doc-7")
    assert 0.45 < halves.count("calib") / len(halves) < 0.55


def log_softmax(logits):
    return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))


def test_binoculars_is_one_when_both_models_agree_and_text_is_typical():
    # With identical models, the ratio compares the text's surprise with the expected surprise.
    rng = np.random.default_rng(1)
    logp = log_softmax(rng.normal(size=(4000, 30)))
    sampled = np.array([rng.choice(30, p=np.exp(row)) for row in logp])
    stats = statistics(logp, logp, sampled)
    assert abs(stats["binoculars"] - 1) < 0.03
    assert abs(stats["fastdetect"]) < 3


def test_predictable_text_scores_as_machine_like():
    rng = np.random.default_rng(2)
    logp = log_softmax(3 * rng.normal(size=(300, 30)))
    greedy = logp.argmax(axis=1)                                 # always the most likely token
    sampled = np.array([rng.choice(30, p=np.exp(row)) for row in logp])
    assert statistics(logp, logp, greedy)["binoculars"] < statistics(logp, logp, sampled)["binoculars"]
    assert statistics(logp, logp, greedy)["fastdetect"] > statistics(logp, logp, sampled)["fastdetect"]


def test_score_of_maps_each_question_to_zero_one():
    assert score_of("ai_yes", {"noul": 0.25}) == 0.25
    assert score_of("author", {"probabilities": {"human": 0.3, "ai": 0.7}}) == 0.7
    assert score_of("pct_ai", {"score": 3.0}) == 0.75
