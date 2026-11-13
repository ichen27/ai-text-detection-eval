"""Score every text in the evaluation set with a Clef decision model on Workers AI.

    python -m detecteval.score clef-flash clef clef-omni [--limit N] [--variant joint|repeat]

Each question is asked in its own request, because answers shift when questions share a request.
Results append to results/scores/<model>.jsonl and finished ids are skipped, so a run can be resumed.

Two variants exist only as checks, written to <model>.<variant>.jsonl:
  joint   all three questions in one request (measures how much they influence each other)
  repeat  the standard protocol run a second time (measures run-to-run variation)
"""
from __future__ import annotations

import argparse
import json

from . import config
from .hub import WorkersAI
from .questions import QUESTIONS, to_scores

BATCH = 12


def load_eval_set() -> list[dict]:
    with open(config.DATA_DIR / "eval_set.jsonl") as handle:
        return [json.loads(line) for line in handle]


def score_model(model: str, texts: list[dict], variant: str | None = None) -> None:
    out_dir = config.RESULTS_DIR / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (f"{model}.{variant}.jsonl" if variant else f"{model}.jsonl")
    done = set()
    if path.exists():
        with open(path) as handle:
            done = {json.loads(line)["id"] for line in handle}
    todo = [text for text in texts if text["id"] not in done]
    workers = WorkersAI()
    question_sets = [QUESTIONS] if variant == "joint" else [{name: spec} for name, spec in QUESTIONS.items()]
    failures = 0
    with open(path, "a") as handle:
        for start in range(0, len(todo), BATCH):
            chunk = todo[start:start + BATCH]
            items = [{"model": f"@cf/cloudflare/{model}",
                      "body": {"model": model, "state": text["text"], "questions": questions}}
                     for text in chunk for questions in question_sets]
            outcomes = workers.run_batch(items)
            for position, text in enumerate(chunk):
                group = outcomes[position * len(question_sets):(position + 1) * len(question_sets)]
                if not all(outcome.get("ok") for outcome in group):
                    failures += 1
                    continue
                answers, tokens = {}, 0
                for outcome in group:
                    for name, answer in outcome["result"]["answers"].items():
                        answers[name] = {key: value for key, value in answer.items() if key != "legend"}
                    tokens += outcome["result"]["usage"]["input_tokens"]
                handle.write(json.dumps({"id": text["id"], "model": model, "scores": to_scores(answers),
                                         "answers": answers, "input_tokens": tokens}) + "\n")
            handle.flush()
            print(f"{model}: {min(start + BATCH, len(todo))}/{len(todo)}", flush=True)
    print(f"{model}: done, {failures} texts failed -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("models", nargs="+", choices=["clef-flash", "clef", "clef-omni"])
    parser.add_argument("--limit", type=int, help="score only the first N texts")
    parser.add_argument("--variant", choices=["joint", "repeat"], help="write a check variant instead")
    args = parser.parse_args()
    config.load_env()
    texts = load_eval_set()[: args.limit]
    for model in args.models:
        score_model(model, texts, args.variant)


if __name__ == "__main__":
    main()
