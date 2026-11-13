"""Fine-tune an encoder to tell human from machine text.

    python -m detecteval.train_classifier raid       # needs the "ml" dependency group
    python -m detecteval.train_classifier mirrors    # continues from the saved "raid" model

"raid" trains on 2023-era machine text only. "mirrors" then updates that model with a small set of
texts from current models, the way a deployed detector is refreshed when new models appear.

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
import shutil
import time

from . import config, train_data
from .metrics import auroc
from .textprep import trim_words

# A small encoder (82M parameters) so that training fits in about 2 GB of memory. Set CLASSIFIER_BASE to
# try a larger one, for example answerdotai/ModernBERT-base, on a machine with memory to spare.
BASE_MODEL = os.environ.get("CLASSIFIER_BASE", "distilbert/distilroberta-base")
MAX_TOKENS = 512            # about 380 words; longer texts are judged on their first 512 tokens
CROP_WORDS = (50, 380)     # 380 words is about the 512 tokens the encoder reads
PAD_STEP = 128
CHECKPOINT_STEPS = 250


def crop(text: str, rng: random.Random) -> str:
    words = rng.randint(*CROP_WORDS)
    return trim_words(text, words, min(words, 40)) or text


def model_dir(recipe: str):
    return config.raw_dir() / "models" / recipe


def main() -> None:
    config.cap_gpu_memory("0.45")
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
    random.Random(0).shuffle(validation)
    if args.limit:
        train, validation = train[: args.limit], validation[: args.limit]
    print(f"{args.recipe}: {len(train)} training and {len(validation)} validation examples on {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    start = BASE_MODEL
    if args.recipe == "mirrors":
        start = model_dir("raid")
        if not (start / "config.json").exists():
            raise SystemExit("train the 'raid' recipe first; 'mirrors' continues from it")
    model = AutoModelForSequenceClassification.from_pretrained(start, num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = math.ceil(len(train) * args.epochs / args.batch)
    warmup = max(1, int(0.06 * steps))

    def lr_at(step: int) -> float:
        return (step + 1) / warmup if step < warmup else max(0.0, (steps - step) / (steps - warmup))

    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)

    def encode(texts: list[str]):
        # Pad to a few fixed lengths: the Apple GPU allocator keeps a buffer for every new tensor shape,
        # so free-form batch lengths make memory grow without bound.
        rows = tokenizer(texts, truncation=True, max_length=MAX_TOKENS)
        longest = max(len(ids) for ids in rows["input_ids"])
        length = min(MAX_TOKENS, PAD_STEP * math.ceil(longest / PAD_STEP))
        return tokenizer.pad(rows, padding="max_length", max_length=length, return_tensors="pt").to(device)

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

    # A checkpoint every CHECKPOINT_STEPS lets an interrupted run continue; the optimizer restarts cold.
    checkpoint = model_dir(args.recipe) / "checkpoint"
    log, first_step = [], 0
    if not args.limit and (checkpoint / "state.json").exists():
        state = json.loads((checkpoint / "state.json").read_text())
        if state["steps"] == steps:
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
            schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at, last_epoch=-1)
            log, first_step = state["log"], state["step"]
            for _ in range(first_step):
                schedule.step()
            print(f"resuming from step {first_step}", flush=True)
    started, running = time.time(), log[-1]["loss"] if log else 0.0
    model.train()
    for step in range(first_step, steps):
        batch = [train[(step * args.batch + n) % len(train)] for n in range(args.batch)]
        inputs = encode([crop(example["text"], rng) for example in batch])
        labels = torch.tensor([example["label"] for example in batch], device=device)
        loss = torch.nn.functional.cross_entropy(model(**inputs).logits.float(), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        optimizer.zero_grad()
        running = loss.item() if step == first_step and not log else 0.98 * running + 0.02 * loss.item()
        if device == "mps" and step % 25 == 0:
            torch.mps.empty_cache()
        if (step + 1) % 100 == 0 or step + 1 == steps:
            entry = {"step": step + 1, "of": steps, "loss": round(running, 4),
                     "minutes": round((time.time() - started) / 60, 1)}
            if (step + 1) % 500 == 0 or step + 1 == steps:
                entry.update(validate())
            log.append(entry)
            print(json.dumps(entry), flush=True)
        if not args.limit and (step + 1) % CHECKPOINT_STEPS == 0 and step + 1 < steps:
            model.save_pretrained(checkpoint)
            (checkpoint / "state.json").write_text(json.dumps({"step": step + 1, "steps": steps, "log": log}))

    if args.limit:
        print("smoke test only; nothing saved")
        return
    out = model_dir(args.recipe)
    out.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(checkpoint, ignore_errors=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    summary = {"recipe": args.recipe, "base_model": BASE_MODEL,
               "started_from": BASE_MODEL if args.recipe == "raid" else "the saved raid model",
               "training_examples": len(train), "validation_examples": len(validation), "epochs": args.epochs,
               "batch": args.batch, "lr": args.lr, "max_tokens": MAX_TOKENS, "final": log[-1], "log": log}
    (out / "training_log.json").write_text(json.dumps(summary, indent=2))
    results = config.RESULTS_DIR / "detectors"
    results.mkdir(parents=True, exist_ok=True)
    (results / f"classifier-{args.recipe}.training.json").write_text(json.dumps(summary, indent=2))
    print("saved ->", out)


if __name__ == "__main__":
    main()
