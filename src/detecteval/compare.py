"""Compare detectors on both evaluation sets with thresholds that were set on other texts.

    python -m detecteval.compare

Every text belongs to a "calib" or a "test" half (texts sharing a source document share a half).
Thresholds, and the weights of combined detectors, come from the calib half. Every number reported
is measured on the test half. Writes results/comparison.json, results/comparison.md and figures.
"""
from __future__ import annotations

import json

import numpy as np

from . import config, metrics
from .build_long_set import split_of
from .generate_mirrors import TRAIN_GENERATORS
from .textprep import rung_text

DETECTORS = {
    "clef": "Clef 27B, percentage question (no training)",
    "clef-flash": "Clef-flash 9B, human-or-AI question (no training)",
    "clef-omni": "Clef-omni, human-or-AI question (no training)",
    "binoculars": "Binoculars, 0.5B model pair (no training)",
    "fastdetect": "Fast-DetectGPT, 0.5B model pair (no training)",
    "classifier-raid": "Classifier trained on RAID (2023 models)",
    "classifier-mirrors": "Classifier trained on RAID + current-model mirrors",
}
COMBINATIONS = {
    "clef+binoculars": ("Clef + Binoculars", ["clef", "binoculars"]),
    "clef+classifier": ("Clef + classifier (mirrors)", ["clef", "classifier-mirrors"]),
    "all-three": ("Clef + Binoculars + classifier (mirrors)", ["clef", "binoculars", "classifier-mirrors"]),
    "local-two": ("Binoculars + classifier (mirrors)", ["binoculars", "classifier-mirrors"]),
}
GROUP_LABELS = {
    "raid_human": "Human: RAID documents",
    "essay_native": "Human: native-speaker student essays",
    "essay_nonnative": "Human: non-native TOEFL essays",
    "forum_2019": "Human: forum posts from 2019",
    "forum_recent": "Human: forum posts after the models' release",
    "toefl_gpt4_polished": "Mixed: TOEFL essays polished by GPT-4",
    "polished_2026": "Mixed: human text polished by a 2026 model",
    "raid_2023": "AI: 2023 chat models (RAID)",
    "raid_paraphrased": "AI: 2023 models, paraphrased (RAID)",
    "fresh_2026": "AI: 2026 models",
    "humanized_2026": "AI: 2026 models, humanized",
}
# Detectors compared head to head. A text counts only if every one of these has scored it, so no detector
# is judged on an easier or harder set of texts than another.
CORE = ["clef", "binoculars", "fastdetect", "classifier-raid", "classifier-mirrors"]
LOCAL = ["binoculars", "fastdetect", "classifier-raid", "classifier-mirrors"]     # need no paid API
SEEN_GENERATORS = {name.rsplit("/", 1)[1] for name in TRAIN_GENERATORS}
SLICES = [("short", 150)] + [("long", rung) for rung in config.RUNGS]


def load_texts() -> dict[str, dict]:
    """Metadata for every (text, length) key."""
    texts = {}
    with open(config.DATA_DIR / "eval_set.jsonl") as handle:
        for row in map(json.loads, handle):
            texts[row["id"] + "@short"] = {
                "id": row["id"], "set": "short", "rung": 150, "label": row["label"], "group": row["group"],
                "generator": row.get("generator", ""), "split": split_of(row.get("source_id", row["id"]))}
    with open(config.DATA_DIR / "long_set.jsonl") as handle:
        for row in map(json.loads, handle):
            for rung in config.RUNGS:
                if rung_text(row["text"], rung, config.RUNG_FILL) is not None:
                    texts[f"{row['id']}@{rung}"] = {
                        "id": row["id"], "set": "long", "rung": rung, "label": row["label"], "group": row["group"],
                        "generator": row.get("generator", ""), "split": row["split"],
                        "rewritten_from": row.get("rewritten_from")}
    return texts


def read_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as handle:
        return [json.loads(line) for line in handle]


def load_scores() -> dict[str, dict[str, float]]:
    """Scores per detector and key; higher always means more likely AI."""
    results = config.RESULTS_DIR
    scores: dict[str, dict[str, float]] = {}
    clef = {row["id"] + "@short": row["scores"]["pct_ai"] for row in read_jsonl(results / "scores" / "clef.jsonl")}
    clef.update({row["key"]: row["score"] for row in read_jsonl(results / "scores_long" / "clef.pct_ai.jsonl")})
    scores["clef"] = clef
    for small in ("clef-flash", "clef-omni"):
        rows = {row["id"] + "@short": row["scores"]["author"]
                for row in read_jsonl(results / "scores" / f"{small}.jsonl")}
        rows.update({row["key"]: row["score"]
                     for row in read_jsonl(results / "scores_long" / f"{small}.author.jsonl")})
        scores[small] = rows
    local = read_jsonl(results / "detectors" / "binoculars.jsonl")
    scores["binoculars"] = {row["key"]: -row["binoculars"] for row in local}     # a low ratio means machine text
    scores["fastdetect"] = {row["key"]: row["fastdetect"] for row in local}
    for recipe in ("raid", "mirrors"):
        scores[f"classifier-{recipe}"] = {row["key"]: row["logit"] for row in
                                          read_jsonl(results / "detectors" / f"classifier-{recipe}.jsonl")}
    return {name: values for name, values in scores.items() if values}


def keys_of(texts: dict, text_set: str, rung: int, scores: dict | None = None,
            required: list[str] | None = None) -> list[str]:
    """Keys of one slice; with ``scores``, only those every required (default: core) detector has scored."""
    keys = [key for key, text in texts.items() if text["set"] == text_set and text["rung"] == rung]
    if scores:
        core = [scores[name] for name in (required or CORE) if name in scores]
        keys = [key for key in keys if all(key in detector for detector in core)]
    return keys


def values(keys: list[str], texts: dict, scores: dict[str, float], **where) -> np.ndarray:
    return np.array([scores[key] for key in keys
                     if key in scores and all(texts[key][field] == value for field, value in where.items())])


def combine(keys: list[str], texts: dict, scores: dict, parts: list[str]) -> dict[str, float]:
    """Logistic regression over the component scores, fitted on the calib half's human and AI texts."""
    usable = [key for key in keys if all(key in scores[part] for part in parts)]
    fit = [key for key in usable if texts[key]["split"] == "calib" and texts[key]["label"] in ("human", "ai")]
    if len({texts[key]["label"] for key in fit}) < 2:
        return {}

    def matrix(rows: list[str]) -> np.ndarray:
        return np.array([[scores[part][key] for part in parts] for key in rows])

    model = metrics.fit_logistic(matrix(fit), np.array([texts[key]["label"] == "ai" for key in fit], float))
    return dict(zip(usable, metrics.predict_logistic(model, matrix(usable)).tolist(), strict=True))


def evaluate(keys: list[str], texts: dict, scores: dict[str, float]) -> dict | None:
    calib_human = values(keys, texts, scores, label="human", split="calib")
    human = values(keys, texts, scores, label="human", split="test")
    ai = values(keys, texts, scores, label="ai", split="test")
    if min(calib_human.size, human.size, ai.size) < 20:
        return None
    out = {"n_calib_human": int(calib_human.size), "n_test_human": int(human.size), "n_test_ai": int(ai.size),
           "auroc": metrics.auroc(human, ai), "auroc_ci95": metrics.bootstrap_ci(metrics.auroc, human, ai, n=1000)}
    for name, target in (("5pct", 0.05), ("1pct", 0.01)):
        rates = metrics.held_out_rates(calib_human, human, ai, target)
        out[f"threshold_{name}"] = rates["threshold"]
        out[f"tpr_{name}"] = rates["tpr"]
        out[f"tpr_{name}_ci95"] = metrics.wilson_interval(int(round(rates["tpr"] * ai.size)), int(ai.size))
        out[f"fpr_{name}"] = rates["fpr"]
        out[f"fpr_{name}_ci95"] = metrics.wilson_interval(int(round(rates["fpr"] * human.size)), int(human.size))
    return out


def flagged(keys: list[str], texts: dict, scores: dict[str, float], threshold: float, **where) -> dict | None:
    group = values(keys, texts, scores, split="test", **where)
    if group.size == 0:
        return None
    count = int((group > threshold).sum())
    return {"n": int(group.size), "flagged": count, "share": count / group.size,
            "ci95": metrics.wilson_interval(count, int(group.size))}


def evaluate_slice(keys: list[str], texts: dict, scores: dict, names: list[str]) -> dict:
    """Overall and per-group results for the named detectors and every combination they allow."""
    # A detector that has scored only part of a slice would be judged on an arbitrary subset, so it is left out.
    local = {name: scores[name] for name in names
             if name in scores and keys and sum(key in scores[name] for key in keys) >= 0.98 * len(keys)}
    for name, (_, parts) in COMBINATIONS.items():
        if all(part in local for part in parts):
            combined = combine(keys, texts, scores, parts)
            if combined:
                local[name] = combined
    entry: dict = {"n_texts": len(keys), "detectors": {}}
    for name, detector_scores in local.items():
        overall = evaluate(keys, texts, detector_scores)
        if overall is None:
            continue
        groups = {}
        for group in GROUP_LABELS:
            cell = flagged(keys, texts, detector_scores, overall["threshold_5pct"], group=group)
            if cell:
                groups[group] = cell
        writers = {}
        for label, seen in (("seen in training", True), ("never seen in training", False)):
            rows = [key for key in keys if texts[key]["group"] == "fresh_2026" and texts[key]["split"] == "test"
                    and key in detector_scores and (texts[key]["generator"] in SEEN_GENERATORS) == seen]
            count = sum(detector_scores[key] > overall["threshold_5pct"] for key in rows)
            writers[label] = {"n": len(rows), "flagged": int(count), "share": count / len(rows) if rows else None,
                              "ci95": metrics.wilson_interval(int(count), len(rows))}
        entry["detectors"][name] = {**overall, "groups": groups, "fresh_2026_by_generator": writers}
    return entry


def analyse() -> dict:
    texts, scores = load_texts(), load_scores()
    out: dict = {"slices": {}, "local_slices": {}, "detectors": {name: DETECTORS[name] for name in scores},
                 "combinations": {name: label for name, (label, _) in COMBINATIONS.items()}}
    for text_set, rung in SLICES:
        entry = evaluate_slice(keys_of(texts, text_set, rung, scores), texts, scores, list(DETECTORS))
        entry["n_in_set"] = len(keys_of(texts, text_set, rung))
        out["slices"][f"{text_set}@{rung}"] = entry
    # The detectors that run locally have scored every long-set text, including ones a paid model has not.
    for rung in config.RUNGS:
        out["local_slices"][f"long@{rung}"] = evaluate_slice(keys_of(texts, "long", rung, scores, LOCAL), texts,
                                                              scores, LOCAL)

    # The 500-word rung covers fewer documents; score the same documents at 300 words for a fair comparison.
    long_500 = {texts[key]["id"] for key in keys_of(texts, "long", 500, scores, LOCAL)}
    same_docs = [key for key in keys_of(texts, "long", 300, scores, LOCAL) if texts[key]["id"] in long_500]
    out["local_slices"]["long@300 (documents that reach 500 words)"] = evaluate_slice(same_docs, texts, scores, LOCAL)

    # Does the model treat human text it may have memorised differently from text it cannot have seen?
    keys = keys_of(texts, "long", 300, scores)
    out["seen_vs_unseen_humans"] = {}
    for name, detector_scores in scores.items():
        older = values(keys, texts, detector_scores, group="forum_2019")
        recent = values(keys, texts, detector_scores, group="forum_recent")
        if older.size and recent.size:
            out["seen_vs_unseen_humans"][name] = {
                "n_2019": int(older.size), "n_recent": int(recent.size),
                "median_2019": float(np.median(older)), "median_recent": float(np.median(recent)),
                "auroc_recent_over_2019": metrics.auroc(older, recent),
                "auroc_ci95": metrics.bootstrap_ci(metrics.auroc, older, recent, n=1000)}

    # Paired change when a model polishes the same human text.
    out["polish_effect"] = {}
    for name, detector_scores in scores.items():
        diffs = [detector_scores[key] - detector_scores[f"{texts[key]['rewritten_from']}@300"] for key in keys
                 if texts[key]["group"] == "polished_2026" and key in detector_scores
                 and f"{texts[key]['rewritten_from']}@300" in detector_scores]
        if diffs:
            out["polish_effect"][name] = {"n_pairs": len(diffs), "share_increased": float(np.mean(np.array(diffs) > 0))}
    return out


def main() -> None:
    from .compare_report import figures, markdown

    summary = analyse()
    (config.RESULTS_DIR / "comparison.json").write_text(json.dumps(summary, indent=1) + "\n")
    (config.RESULTS_DIR / "comparison.md").write_text(markdown(summary))
    figures(summary)
    print((config.RESULTS_DIR / "comparison.md").read_text())


if __name__ == "__main__":
    main()
