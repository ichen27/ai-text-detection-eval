"""Reference score distributions that turn a raw detector score into "how unusual is this for human text".

    python -m detecteval.calibration            # public reference from the committed long-set scores
    python -m detecteval.calibration --private  # also a private human reference, kept outside the repository

The public file (results/calibration.json) holds, for each detector and text length, the sorted scores of
every human and every AI text in the long set. The optional private file
(RAW_DATA_DIR/private/calibration.json) holds the scores of one group of private human documents, by
default the writer's pre-ChatGPT work, which is the harder and more realistic comparison.
"""
from __future__ import annotations

import argparse
import bisect
import json

from . import compare, config
from .items import private_dir, private_documents

DETECTORS = ["classifier-mirrors", "binoculars"]
PRIVATE_GROUP = "own_pre_ai"


def public_path():
    return config.RESULTS_DIR / "calibration.json"


def private_path():
    return private_dir() / "calibration.json"


def build_public() -> dict:
    texts, scores = compare.load_texts(), compare.load_scores()
    out: dict = {}
    for name in DETECTORS:
        out[name] = {}
        for rung in config.RUNGS:
            keys = [key for key in compare.keys_of(texts, "long", rung) if key in scores[name]]
            out[name][str(rung)] = {
                label: sorted(round(scores[name][key], 4) for key in keys if texts[key]["label"] == label)
                for label in ("human", "ai")}
    return out


def build_private(group: str = PRIVATE_GROUP) -> dict:
    from .private_eval import load_private_scores

    scores = load_private_scores()
    documents = [row["id"] for row in private_documents() if row["group"] == group]
    return {name: {str(rung): {"human": sorted(round(scores[name][f"{doc}@{rung}"], 4) for doc in documents
                                               if f"{doc}@{rung}" in scores[name])}
                   for rung in config.RUNGS}
            for name in DETECTORS if name in scores}


def load() -> dict:
    """{"public": ..., "private": ... or None}. Raises if the public reference has not been built."""
    with open(public_path()) as handle:
        public = json.load(handle)
    private = None
    if private_path().exists():
        with open(private_path()) as handle:
            private = json.load(handle)
    return {"public": public, "private": private}


def share_at_least(reference: list[float], score: float) -> float | None:
    """Share of the sorted reference scores that are at least ``score``; None without a reference."""
    if not reference:
        return None
    return (len(reference) - bisect.bisect_left(reference, score)) / len(reference)


def share_at_most(reference: list[float], score: float) -> float | None:
    if not reference:
        return None
    return bisect.bisect_right(reference, score) / len(reference)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--private", action="store_true", help="also write the private human reference")
    args = parser.parse_args()
    public_path().write_text(json.dumps(build_public()) + "\n")
    print("wrote", public_path())
    if args.private:
        private_path().write_text(json.dumps(build_private()) + "\n")
        print("wrote", private_path())


if __name__ == "__main__":
    main()
