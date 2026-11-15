"""Test the detectors on private writing, such as your own unpublished documents.

Public human text may be in a model's training data, and recent public text may be AI-assisted.
Documents that were never online and were written before language models were common are the clean
test of how often a detector accuses a real person.

    python -m detecteval.private_eval clef      # paid: sends each text to Clef 27B on Workers AI
    python -m detecteval.private_eval local     # Binoculars and both classifiers; needs the "ml" group
    python -m detecteval.private_eval report    # writes results/private_writing.md and .json

Put the documents in RAW_DATA_DIR/private/manifest.jsonl, one JSON object per line:

    {"id": "doc-001", "group": "own_pre_ai", "text": "...", "modified": "2021-04-02"}

``group`` is free text; every group is reported separately. The texts and the per-document scores stay
under RAW_DATA_DIR/private, outside the repository. The report holds counts and rates only: no text,
no titles and no document ids. The flagging bar and the combination weights are the ones set on the
calibration half of the long set, so the private documents play no part in choosing them.
"""
from __future__ import annotations

import argparse
import collections
import json

import numpy as np

from . import compare, config, metrics
from .items import private_dir, private_documents, private_items

DETECTORS = ["clef", "binoculars", "classifier-raid", "classifier-mirrors"]
COMBINATIONS = {"clef+binoculars": ["clef", "binoculars"],
                "all-three": ["clef", "binoculars", "classifier-mirrors"]}
NAMES = {"clef": "Clef 27B, asked", "binoculars": "Binoculars (0.5B pair)",
         "classifier-raid": "Classifier, RAID only", "classifier-mirrors": "Classifier, RAID + mirrors",
         "clef+binoculars": "Clef + Binoculars", "all-three": "All three combined"}
GROUP_NAMES = {"own_pre_ai": "Written before ChatGPT (30 Nov 2022)",
               "own_recent": "Written since, AI use unknown"}


def scores_dir():
    return private_dir() / "scores"


def load_private_scores() -> dict[str, dict[str, float]]:
    """Scores per detector and key, oriented like ``compare.load_scores`` (higher = more likely AI)."""
    directory = scores_dir()
    local = compare.read_jsonl(directory / "binoculars.jsonl")
    scores = {
        "clef": {row["key"]: row["score"] for row in compare.read_jsonl(directory / "clef.pct_ai.jsonl")},
        "binoculars": {row["key"]: -row["binoculars"] for row in local},
        "classifier-raid": {row["key"]: row["logit"] for row in
                            compare.read_jsonl(directory / "classifier-raid.jsonl")},
        "classifier-mirrors": {row["key"]: row["logit"] for row in
                               compare.read_jsonl(directory / "classifier-mirrors.jsonl")},
    }
    return {name: values for name, values in scores.items() if values}


def rate(values: np.ndarray, threshold: float) -> dict:
    count = int((values > threshold).sum())
    return {"n": int(values.size), "flagged": count, "share": count / values.size if values.size else None,
            "ci95": metrics.wilson_interval(count, int(values.size))}


def analyse(target_fpr: float = 0.05) -> dict:
    texts, public = compare.load_texts(), compare.load_scores()
    private = load_private_scores()
    groups = {row["id"]: row["group"] for row in private_documents()}
    out: dict = {"target_fpr": target_fpr, "documents": dict(collections.Counter(groups.values())), "rungs": {}}
    for rung in config.RUNGS:
        keys = compare.keys_of(texts, "long", rung, public)
        calib = [key for key in keys if texts[key]["split"] == "calib" and texts[key]["label"] in ("human", "ai")]
        calib_human = [key for key in calib if texts[key]["label"] == "human"]
        test_human = [key for key in keys if texts[key]["split"] == "test" and texts[key]["label"] == "human"]
        mine = collections.defaultdict(list)
        for document, group in groups.items():
            key = f"{document}@{rung}"
            if all(key in private.get(name, {}) for name in DETECTORS):
                mine[group].append(key)
        if len(calib_human) < 20 or not mine:
            continue
        entry: dict = {}
        plans = {name: [name] for name in DETECTORS} | COMBINATIONS
        for name, parts in plans.items():
            if not all(part in public and part in private for part in parts):
                continue
            if len(parts) == 1:
                def score(rows, source, part=parts[0]):
                    return np.array([source[part][key] for key in rows])
            else:
                # Weights come from the public calibration half only.
                def matrix(rows, source, parts=parts):
                    return np.array([[source[part][key] for part in parts] for key in rows])
                model = metrics.fit_logistic(matrix(calib, public),
                                             np.array([texts[key]["label"] == "ai" for key in calib], float))

                def score(rows, source, model=model, matrix=matrix):
                    return metrics.predict_logistic(model, matrix(rows, source))
            threshold = metrics.threshold_for_fpr(score(calib_human, public), target_fpr)
            entry[name] = {"public_test_humans": rate(score(test_human, public), threshold),
                           "groups": {group: rate(score(rows, private), threshold) for group, rows in mine.items()}}
        out["rungs"][str(rung)] = entry
    return out


def markdown(summary: dict) -> str:
    lines = ["# Private writing", "",
             "How often each detector flags documents that were never published. The flagging bar is the one "
             f"set to flag {summary['target_fpr']:.0%} of human texts on the calibration half of the long set; "
             "these documents had no part in setting it. A detector that treats this writing like other human "
             f"writing should flag about {summary['target_fpr']:.0%} of it.", "",
             "Documents: " + ", ".join(f"{GROUP_NAMES.get(group, group)}: {count}"
                                       for group, count in sorted(summary["documents"].items())) + ".",
             "The texts, titles and per-document scores are not in this repository."]
    for rung, entry in summary["rungs"].items():
        groups = sorted({group for detector in entry.values() for group in detector["groups"]})
        lines += ["", f"## At {rung} words", "",
                  "| Detector | Public human texts (test half) | "
                  + " | ".join(GROUP_NAMES.get(group, group) for group in groups) + " |",
                  "|---|---|" + "---|" * len(groups)]
        for name, detector in entry.items():
            cells = []
            for cell in [detector["public_test_humans"]] + [detector["groups"].get(group) for group in groups]:
                cells.append("" if not cell or not cell["n"] else
                             f"{cell['flagged']} of {cell['n']} ({cell['share']:.0%}; "
                             f"{cell['ci95'][0]:.0%}-{cell['ci95'][1]:.0%})")
            lines.append(f"| {NAMES[name]} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=["clef", "local", "report"])
    args = parser.parse_args()
    config.load_env()
    if args.step == "clef":
        from .score_long import score
        score("clef", "pct_ai", config.RUNGS, private_items(), scores_dir() / "clef.pct_ai.jsonl")
    elif args.step == "local":
        from . import binoculars, classify
        binoculars.run(private_items(), scores_dir() / "binoculars.jsonl")
        classify.run(["raid", "mirrors"], private_items(), scores_dir())
    else:
        summary = analyse()
        (config.RESULTS_DIR / "private_writing.json").write_text(json.dumps(summary, indent=1) + "\n")
        (config.RESULTS_DIR / "private_writing.md").write_text(markdown(summary))
        print(markdown(summary))


if __name__ == "__main__":
    main()
