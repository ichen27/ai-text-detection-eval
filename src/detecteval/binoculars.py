"""Binoculars and Fast-DetectGPT: detectors that measure how predictable a text is to small language models.

Both need two related models, an "observer" and a "performer", and no training.

  Binoculars (Hans et al., 2024): the performer's log-perplexity on the text, divided by the
  cross-entropy between the observer's and the performer's next-token predictions. Machine text
  is unsurprising relative to what the models expect, so it gets a LOW ratio.

  Fast-DetectGPT (Bao et al., 2024), analytic form: how far the text's log-likelihood under the
  performer lies above the mean log-likelihood of tokens sampled from the observer, in standard
  deviations. Machine text gets a HIGH value.

The original papers use 7B-parameter models. This runs a 0.5B pair so that it fits in a few gigabytes
of memory, which costs accuracy; the numbers here are a floor for the method, not its ceiling.
Set BINOCULARS_OBSERVER and BINOCULARS_PERFORMER to use a larger pair that shares one tokenizer.

    python -m detecteval.binoculars            # needs the "ml" dependency group

Appends to results/detectors/binoculars.jsonl; finished keys are skipped.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import config
from .items import all_items, done_keys

OBSERVER = os.environ.get("BINOCULARS_OBSERVER", "Qwen/Qwen2.5-0.5B")
PERFORMER = os.environ.get("BINOCULARS_PERFORMER", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_TOKENS = 768
BATCH = 1   # a batch of full-vocabulary logits is large; one text at a time keeps memory flat


def statistics(observer_logp: np.ndarray, performer_logp: np.ndarray, tokens: np.ndarray) -> dict:
    """Detector statistics for one text.

    ``observer_logp`` and ``performer_logp`` are log-probabilities of shape (positions, vocabulary)
    for predicting ``tokens`` (shape (positions,)), i.e. row i predicts tokens[i].
    """
    observer_p = np.exp(observer_logp)
    log_likelihood = performer_logp[np.arange(len(tokens)), tokens]
    log_ppl = -log_likelihood.mean()
    cross_entropy = -(observer_p * performer_logp).sum(axis=1)
    mean_ref = -cross_entropy
    var_ref = (observer_p * performer_logp ** 2).sum(axis=1) - mean_ref ** 2
    discrepancy = (log_likelihood.sum() - mean_ref.sum()) / np.sqrt(max(var_ref.sum(), 1e-9))
    return {"binoculars": float(log_ppl / cross_entropy.mean()), "fastdetect": float(discrepancy),
            "log_ppl": float(log_ppl), "tokens": int(len(tokens))}


POSITIONS_PER_CHUNK = 96   # full-vocabulary tensors are large; working through a text in slices bounds memory


def statistics_torch(observer_logits, performer_logits, tokens) -> dict:
    """The same statistics as ``statistics``, computed on the GPU a slice of positions at a time.

    Takes raw logits of shape (positions, vocabulary); row i predicts tokens[i]."""
    import torch

    log_likelihood = mean_ref = var_ref = 0.0
    for start in range(0, tokens.shape[0], POSITIONS_PER_CHUNK):
        rows = slice(start, start + POSITIONS_PER_CHUNK)
        observer_logp = torch.log_softmax(observer_logits[rows].float(), dim=-1)
        performer_logp = torch.log_softmax(performer_logits[rows].float(), dim=-1)
        observer_p = observer_logp.exp()
        log_likelihood += float(performer_logp.gather(1, tokens[rows, None]).sum())
        expected = (observer_p * performer_logp).sum(dim=1)
        mean_ref += float(expected.sum())
        var_ref += float(((observer_p * performer_logp ** 2).sum(dim=1) - expected ** 2).sum())
    count = int(tokens.shape[0])
    return {"binoculars": log_likelihood / mean_ref,
            "fastdetect": (log_likelihood - mean_ref) / max(var_ref, 1e-9) ** 0.5,
            "log_ppl": -log_likelihood / count, "tokens": count}


def run(items: list[dict], path) -> None:
    """Score ``items`` and append to ``path``; keys already in the file are skipped."""
    config.cap_gpu_memory("0.45")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = config.torch_device()
    dtype = torch.float32 if device == "cpu" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(OBSERVER)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    models = [AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()
              for name in (OBSERVER, PERFORMER)]

    path.parent.mkdir(parents=True, exist_ok=True)
    finished = done_keys(path)
    todo = sorted((item for item in items if item["key"] not in finished), key=lambda item: len(item["text"]))
    with open(path, "a") as handle, torch.no_grad():
        for start in range(0, len(todo), BATCH):
            chunk = todo[start:start + BATCH]
            batch = tokenizer([item["text"] for item in chunk], return_tensors="pt", padding=True,
                              truncation=True, max_length=MAX_TOKENS).to(device)
            logits = [model(**batch).logits for model in models]
            lengths = batch["attention_mask"].sum(dim=1).tolist()
            for row, (item, length) in enumerate(zip(chunk, lengths, strict=True)):
                # Row i of the logits predicts token i + 1; the first token has no prediction.
                stats = statistics_torch(logits[0][row, :length - 1], logits[1][row, :length - 1],
                                         batch["input_ids"][row, 1:length])
                handle.write(json.dumps({"key": item["key"], "id": item["id"], "set": item["set"],
                                         "rung": item["rung"], **stats}) + "\n")
            del logits
            if device == "mps":
                torch.mps.empty_cache()
            handle.flush()
            if (start // BATCH) % 200 == 0:
                print(f"{min(start + BATCH, len(todo))}/{len(todo)}", flush=True)
    print("done ->", path)


def main() -> None:
    run(all_items(), config.RESULTS_DIR / "detectors" / "binoculars.jsonl")


if __name__ == "__main__":
    main()
