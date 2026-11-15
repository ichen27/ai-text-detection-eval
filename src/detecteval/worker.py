"""Model process for the detector service: reads one JSON request per line, writes one JSON reply per line.

    {"texts": ["..."], "binoculars": false}  ->  {"classifier-mirrors": [..], "binoculars": [..] | null}

The service starts this process on demand and stops it after a quiet period, so the models occupy
memory only while someone is using the detector. The classifier runs on the CPU (small and fast
enough); the Binoculars pair loads on the first request that asks for it, on the GPU when there is one.
Scores match the ones in the evaluation: same models, same token limits, same padding.
"""
from __future__ import annotations

import json
import sys

from . import config
from .binoculars import MAX_TOKENS as BINOCULARS_TOKENS
from .binoculars import OBSERVER, PERFORMER, statistics_torch
from .classify import MAX_TOKENS as CLASSIFIER_TOKENS
from .train_classifier import model_dir

RECIPE = "mirrors"


class Models:
    def __init__(self) -> None:
        config.cap_gpu_memory("0.35")
        import torch

        self.torch = torch
        self.classifier = None
        self.pair = None

    def classifier_logits(self, texts: list[str]) -> list[float]:
        torch = self.torch
        if self.classifier is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            path = model_dir(RECIPE)
            self.classifier = (AutoTokenizer.from_pretrained(path),
                               AutoModelForSequenceClassification.from_pretrained(path).eval())
        tokenizer, model = self.classifier
        out = []
        with torch.no_grad():
            for text in texts:
                inputs = tokenizer([text], return_tensors="pt", padding="max_length", truncation=True,
                                   max_length=CLASSIFIER_TOKENS)
                logits = model(**inputs).logits.float()
                out.append(float(logits[0, 1] - logits[0, 0]))
        return out

    def binoculars_scores(self, texts: list[str]) -> list[float]:
        torch = self.torch
        if self.pair is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = config.torch_device()
            dtype = torch.float32 if device == "cpu" else torch.float16
            tokenizer = AutoTokenizer.from_pretrained(OBSERVER)
            models = [AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()
                      for name in (OBSERVER, PERFORMER)]
            self.pair = (tokenizer, models, device)
        tokenizer, models, device = self.pair
        out = []
        with torch.no_grad():
            for text in texts:
                batch = tokenizer([text], return_tensors="pt", truncation=True, max_length=BINOCULARS_TOKENS).to(device)
                logits = [model(**batch).logits for model in models]
                length = int(batch["attention_mask"].sum())
                stats = statistics_torch(logits[0][0, :length - 1], logits[1][0, :length - 1],
                                         batch["input_ids"][0, 1:length])
                out.append(-stats["binoculars"])          # a low ratio means machine text
                del logits
                if device == "mps":
                    torch.mps.empty_cache()
        return out


def main() -> None:
    config.load_env()
    models = Models()
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            reply = {"classifier-mirrors": models.classifier_logits(request["texts"]),
                     "binoculars": models.binoculars_scores(request["texts"]) if request.get("binoculars") else None}
        except Exception as error:  # noqa: BLE001 - the service needs the reason, whatever it is
            reply = {"error": f"{type(error).__name__}: {error}"[:300]}
        print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    main()
