"""Score the long set with a Clef decision model, one question per request, at chosen lengths.

    python -m detecteval.score_long clef --question pct_ai --rungs 75 150 300 500

Appends to results/scores_long/<model>.<question>.jsonl; finished keys are skipped on a rerun.
"""
from __future__ import annotations

import argparse
import json

from . import config
from .hub import WorkersAI
from .items import done_keys, long_items
from .questions import QUESTIONS, score_of

BATCH = 12


def score(model: str, question: str, rungs: list[int], items: list[dict] | None = None, path=None) -> None:
    """Score the long set, or ``items`` when given, appending to ``path``; finished keys are skipped."""
    path = path or config.RESULTS_DIR / "scores_long" / f"{model}.{question}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    finished = done_keys(path)
    todo = [item for item in (long_items(rungs) if items is None else items) if item["key"] not in finished]
    workers, failures = WorkersAI(), 0
    with open(path, "a") as handle:
        for start in range(0, len(todo), BATCH):
            chunk = todo[start:start + BATCH]
            items = [{"model": f"@cf/cloudflare/{model}",
                      "body": {"model": model, "state": item["text"], "questions": {question: QUESTIONS[question]}}}
                     for item in chunk]
            for item, outcome in zip(chunk, workers.run_batch(items), strict=True):
                if not outcome.get("ok"):
                    failures += 1
                    continue
                answer = {key: value for key, value in outcome["result"]["answers"][question].items()
                          if key != "legend"}
                handle.write(json.dumps({
                    "key": item["key"], "id": item["id"], "rung": item["rung"], "model": model,
                    "score": score_of(question, answer), "answer": answer,
                    "input_tokens": outcome["result"]["usage"]["input_tokens"]}) + "\n")
            handle.flush()
            print(f"{model} {question}: {min(start + BATCH, len(todo))}/{len(todo)}", flush=True)
    print(f"{model} {question}: done, {failures} failed -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("models", nargs="+", choices=["clef-flash", "clef", "clef-omni"])
    parser.add_argument("--question", default="pct_ai", choices=sorted(QUESTIONS))
    parser.add_argument("--rungs", nargs="+", type=int, default=config.RUNGS)
    args = parser.parse_args()
    config.load_env()
    for model in args.models:
        score(model, args.question, args.rungs)


if __name__ == "__main__":
    main()
