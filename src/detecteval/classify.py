"""Run a trained classifier over every evaluation text.

    python -m detecteval.classify raid mirrors      # needs the "ml" dependency group

Writes results/detectors/classifier-<recipe>.jsonl (rewritten on each run).
"""
from __future__ import annotations

import argparse
import json

from . import config
from .items import all_items
from .train_classifier import model_dir

MAX_TOKENS = 512      # the encoder's limit, about 380 words; longer texts are judged on their start
BATCH = 8


def main() -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("recipes", nargs="+", choices=["raid", "mirrors"])
    args = parser.parse_args()
    device = config.torch_device()
    items = sorted(all_items(), key=lambda item: len(item["text"]))
    for recipe in args.recipes:
        tokenizer = AutoTokenizer.from_pretrained(model_dir(recipe))
        model = AutoModelForSequenceClassification.from_pretrained(model_dir(recipe)).to(device).eval()
        path = config.RESULTS_DIR / "detectors" / f"classifier-{recipe}.jsonl"
        rows = []
        with torch.no_grad():
            for start in range(0, len(items), BATCH):
                chunk = items[start:start + BATCH]
                inputs = tokenizer([item["text"] for item in chunk], return_tensors="pt", padding=True,
                                   truncation=True, max_length=MAX_TOKENS).to(device)
                logits = model(**inputs).logits.float()
                # The log-odds keep more resolution than a probability that saturates at 1.0.
                margin = (logits[:, 1] - logits[:, 0]).cpu().tolist()
                rows += [{"key": item["key"], "id": item["id"], "set": item["set"], "rung": item["rung"],
                          "logit": round(value, 4)} for item, value in zip(chunk, margin, strict=True)]
        rows.sort(key=lambda row: row["key"])
        with open(path, "w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        print(recipe, len(rows), "->", path)


if __name__ == "__main__":
    main()
