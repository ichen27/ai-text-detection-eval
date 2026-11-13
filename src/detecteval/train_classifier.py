"""Fine-tune an encoder to tell human from machine text.

    python -m detecteval.train_classifier raid       # needs the "ml" dependency group
    python -m detecteval.train_classifier mirrors

Every training text is cut to a random length between 50 and 450 words each time it is seen, so the
model learns to judge short and long texts alike. The model is saved under RAW_DATA_DIR/models/<recipe>
with a small training log; weights are not kept in Git.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

from . import config, train_data
from .metrics import auroc
from .textprep import trim_words

# A small encoder (82M parameters) so that training fits in about 2 GB of memory. Set CLASSIFIER_BASE to
# try a larger one, for example answerdotai/ModernBERT-base, on a machine with memory to spare.
BASE_MODEL = os.environ.get("CLASSIFIER_BASE", "distilbert/distilroberta-base")
MAX_TOKENS = 512            # about 380 words; longer texts are judged on their first 512 tokens
CROP_WORDS = (50, 450)


def crop(text: str, rng: random.Random) -> str:
    words = rng.randint(*CROP_WORDS)
    return trim_words(text, words, min(words, 40)) or text


def model_dir(recipe: str):
    return config.raw_dir() / "models" / recipe


def main() -> None:
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("recipe", choices=["raid", "mirrors"])
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--limit", type=int, help="train on the first N examples only (smoke test)")
    args = parser.parse_args()

    torch.manual_seed(config.SEED)
    rng = random.Random(config.SEED + 40)
    device = config.torch_device()
    train, validation = train_data.load(args.recipe)
    rng.shuffle(train)
    if args.limit:
        train, validation = train[: args.limit], validation[: args.limit]
    print(f"{args.recipe}: {len(train)} training and {len(validation)} validation examples on {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = math.ceil(len(train) * args.epochs / args.batch)
    warmup = max(1, int(0.06 * steps))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: (step + 1) / warmup if step < warmup else max(0.0, (steps - step) / (steps - warmup)))

    def encode(texts: list[str]):
        return tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TOKENS).to(device)

    def validate() -> dict:
        model.eval()
        scores = []
        check = random.Random(0)
        texts = [crop(example["text"], check) for example in validation]
        with torch.no_grad():
            for start in range(0, len(texts), 32):
                logits = model(**encode(texts[start:start + 32])).logits.float()
                scores += torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
        model.train()
        scores, labels = np.array(scores), np.array([example["label"] for example in validation])
        return {"auroc": round(auroc(scores[labels == 0], scores[labels == 1]), 4),
                "accuracy": round(float(((scores > 0.5) == labels).mean()), 4)}

    log, started, running = [], time.time(), 0.0
    model.train()
    for step in range(steps):
        batch = [train[(step * args.batch + n) % len(train)] for n in range(args.batch)]
        inputs = encode([crop(example["text"], rng) for example in batch])
        labels = torch.tensor([example["label"] for example in batch], device=device)
        loss = torch.nn.functional.cross_entropy(model(**inputs).logits.float(), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        optimizer.zero_grad()
        running = loss.item() if step == 0 else 0.98 * running + 0.02 * loss.item()
        if (step + 1) % 100 == 0 or step + 1 == steps:
            entry = {"step": step + 1, "of": steps, "loss": round(running, 4),
                     "minutes": round((time.time() - started) / 60, 1)}
            if (step + 1) % 500 == 0 or step + 1 == steps:
                entry.update(validate())
            log.append(entry)
            print(json.dumps(entry), flush=True)

    if args.limit:
        print("smoke test only; nothing saved")
        return
    out = model_dir(args.recipe)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    summary = {"recipe": args.recipe, "base_model": BASE_MODEL, "training_examples": len(train),
               "validation_examples": len(validation), "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
               "max_tokens": MAX_TOKENS, "final": log[-1], "log": log}
    (out / "training_log.json").write_text(json.dumps(summary, indent=2))
    results = config.RESULTS_DIR / "detectors"
    results.mkdir(parents=True, exist_ok=True)
    (results / f"classifier-{args.recipe}.training.json").write_text(json.dumps(summary, indent=2))
    print("saved ->", out)


if __name__ == "__main__":
    main()
