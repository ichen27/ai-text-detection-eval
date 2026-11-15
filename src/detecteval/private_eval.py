"""Test the detectors on private writing, such as your own unpublished documents.

Public human text may be in a model's training data, and recent public text may be AI-assisted.
Documents that were never online and were written before language models were common are the clean
test of how often a detector accuses a real person.

    python -m detecteval.private_eval clef      # paid: sends each text to Clef 27B on Workers AI
                                                # (--skip-group NAME keeps a group on this machine)
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
COMBINATIONS = {"local-two": ["binoculars", "classifier-mirrors"],
                "clef+binoculars": ["clef", "binoculars"],
                "all-three": ["clef", "binoculars", "classifier-mirrors"]}
NAMES = {"clef": "Clef 27B, asked", "binoculars": "Binoculars (0.5B pair)",
         "classifier-raid": "Classifier, RAID only", "classifier-mirrors": "Classifier, RAID + mirrors",
         "local-two": "Binoculars + classifier (mirrors)", "clef+binoculars": "Clef + Binoculars",
         "all-three": "All three combined"}
REFERENCE_GROUP = "own_pre_ai"      # the group treated as certainly human when re-setting the bar
GROUP_NAMES = {"own_pre_ai": "Dated before ChatGPT (30 Nov 2022)",
               "own_school_undated": "School work, undated (2019 to mid-2023)",
               "own_journal_2024": "Personal journal, 2024",
               "own_recent": "Dated since, AI use unknown"}


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


def rate(values: np.ndarray, threshold: float, documents: list | None = None) -> dict:
    """Texts over the bar. With ``documents`` (one label per text), also how many documents have any."""
    over = values > threshold
    count = int(over.sum())
    out = {"n": int(values.size), "flagged": count, "share": count / values.size if values.size else None,
           "ci95": metrics.wilson_interval(count, int(values.size))}
    if documents is not None:
        out["documents"] = len(set(documents))
        out["documents_flagged"] = len({doc for doc, hit in zip(documents, over, strict=True) if hit})
    return out


def analyse(target_fpr: float = 0.05) -> dict:
    texts, public = compare.load_texts(), compare.load_scores()
    private = load_private_scores()
    manifest = private_documents()
    groups = {row["id"]: row["group"] for row in manifest}
    source = {row["id"]: row.get("doc", row["id"]) for row in manifest}     # passages cut from one document
    out: dict = {"target_fpr": target_fpr, "passages": dict(collections.Counter(groups.values())),
                 "documents": {group: len({source[doc] for doc, g in groups.items() if g == group})
                               for group in set(groups.values())}, "rungs": {}}
    for rung in config.RUNGS:
        keys = compare.keys_of(texts, "long", rung, public)
        calib = [key for key in keys if texts[key]["split"] == "calib" and texts[key]["label"] in ("human", "ai")]
        calib_human = [key for key in calib if texts[key]["label"] == "human"]
        test_human = [key for key in keys if texts[key]["split"] == "test" and texts[key]["label"] == "human"]
        if len(calib_human) < 20:
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
            # A group appears for a detector only where every one of its documents at this length was scored.
            mine = collections.defaultdict(list)
            for document, group in groups.items():
                key = f"{document}@{rung}"
                if all(key in private[part] for part in parts):
                    mine[group].append(key)
            if not mine:
                continue
            threshold = metrics.threshold_for_fpr(score(calib_human, public), target_fpr)
            entry[name] = {"public_test_humans": rate(score(test_human, public), threshold),
                           "groups": {group: rate(score(rows, private), threshold,
                                                  [source[key.rsplit("@", 1)[0]] for key in rows])
                                      for group, rows in mine.items()}}
        if entry:
            out["rungs"][str(rung)] = entry

    # What each detector catches if the bar is set on the certainly-human private group instead of on
    # public human text. The bar is set and the private rate measured on the same passages, so this is
    # indicative; the AI rate is measured on the public test half.
    out["recalibrated"] = {}
    for rung in config.RUNGS:
        keys = compare.keys_of(texts, "long", rung, public)
        test_ai = [key for key in keys if texts[key]["split"] == "test" and texts[key]["label"] == "ai"]
        test_human = [key for key in keys if texts[key]["split"] == "test" and texts[key]["label"] == "human"]
        calib_human = [key for key in keys if texts[key]["split"] == "calib" and texts[key]["label"] == "human"]
        entry = {}
        for name in DETECTORS:
            if name not in public or name not in private:
                continue
            reference = np.array([private[name][f"{doc}@{rung}"] for doc, group in groups.items()
                                  if group == REFERENCE_GROUP and f"{doc}@{rung}" in private[name]])
            if reference.size < 30 or len(test_ai) < 20:
                continue
            public_bar = metrics.threshold_for_fpr(np.array([public[name][key] for key in calib_human]), target_fpr)
            private_bar = metrics.threshold_for_fpr(reference, target_fpr)
            ai = np.array([public[name][key] for key in test_ai])
            human = np.array([public[name][key] for key in test_human])
            entry[name] = {"reference_passages": int(reference.size),
                           "ai_caught_public_bar": metrics.flag_rate(ai, public_bar),
                           "ai_caught_private_bar": metrics.flag_rate(ai, private_bar),
                           "public_humans_flagged_private_bar": metrics.flag_rate(human, private_bar),
                           "auroc_public_ai_over_reference": metrics.auroc(reference, ai),
                           "auroc_public_ai_over_public_humans": metrics.auroc(human, ai)}
        if entry:
            out["recalibrated"][str(rung)] = entry
    return out


def markdown(summary: dict) -> str:
    lines = ["# Private writing", "",
             "How often each detector flags documents that were never published. The flagging bar is the one "
             f"set to flag {summary['target_fpr']:.0%} of human texts on the calibration half of the long set; "
             "these documents had no part in setting it. A detector that treats this writing like other human "
             f"writing should flag about {summary['target_fpr']:.0%} of it.", "",
             "Long documents are cut into consecutive passages of up to 500 words, so passages from one "
             "document are not independent. Cells read: passages flagged of passages scored (share; 95% "
             "interval treating passages as independent), then documents with at least one flagged passage.", "",
             "| Group | Documents | Passages |", "|---|---|---|"]
    lines += [f"| {GROUP_NAMES.get(group, group)} | {summary['documents'][group]} | {count} |"
              for group, count in sorted(summary["passages"].items())]
    lines += ["", "The texts, titles and per-document scores are not in this repository. A blank cell means the "
              "detector was not run on that group."]
    for rung, entry in summary["rungs"].items():
        groups = sorted({group for detector in entry.values() for group in detector["groups"]})
        lines += ["", f"## At {rung} words", "",
                  "| Detector | Public human texts (test half) | "
                  + " | ".join(GROUP_NAMES.get(group, group) for group in groups) + " |",
                  "|---|---|" + "---|" * len(groups)]
        for name, detector in entry.items():
            cells = []
            for cell in [detector["public_test_humans"]] + [detector["groups"].get(group) for group in groups]:
                if not cell or not cell["n"]:
                    cells.append("")
                    continue
                text = (f"{cell['flagged']} of {cell['n']} ({cell['share']:.0%}; "
                        f"{cell['ci95'][0]:.0%}-{cell['ci95'][1]:.0%})")
                if "documents" in cell:
                    text += f"; {cell['documents_flagged']} of {cell['documents']} documents"
                cells.append(text)
            lines.append(f"| {NAMES[name]} | " + " | ".join(cells) + " |")
    if summary.get("recalibrated"):
        reference = GROUP_NAMES.get(REFERENCE_GROUP, REFERENCE_GROUP)
        lines += ["", f"## If the bar were set on the private human group (\"{reference}\")", "",
                  f"The bar is moved to flag {summary['target_fpr']:.0%} of that group's passages, and AI text "
                  "caught is measured on the public test half. The bar is set on the same private passages it "
                  "is judged on, so treat these as indicative. AUROC compares public AI text with each kind of "
                  "human text; 1.0 is perfect separation.", "",
                  "| Length | Detector | AI caught, bar set on public humans | AI caught, bar set on private humans | "
                  "AUROC against public humans | AUROC against private humans |", "|---|---|---|---|---|---|"]
        for rung, entry in summary["recalibrated"].items():
            for name, row in entry.items():
                lines.append(f"| {rung} words | {NAMES[name]} | {row['ai_caught_public_bar']:.0%} | "
                             f"{row['ai_caught_private_bar']:.0%} | {row['auroc_public_ai_over_public_humans']:.2f} | "
                             f"{row['auroc_public_ai_over_reference']:.2f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=["clef", "local", "report"])
    parser.add_argument("--skip-group", action="append", default=[],
                        help="with 'clef': keep this group's texts on this machine (may be repeated)")
    args = parser.parse_args()
    config.load_env()
    if args.step == "clef":
        from .score_long import score
        held_back = {row["id"] for row in private_documents() if row["group"] in args.skip_group}
        items = [item for item in private_items() if item["id"] not in held_back]
        score("clef", "pct_ai", config.RUNGS, items, scores_dir() / "clef.pct_ai.jsonl")
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
