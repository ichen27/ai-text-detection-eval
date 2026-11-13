"""Shared paths and experiment constants."""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "data"
RESULTS_DIR = REPO / "results"
SEED = 20261113
MAX_WORDS = 150
MIN_WORDS = 60

RAID_DOMAINS = ["abstracts", "books", "news", "poetry"]
RAID_CHAT_MODELS = ["chatgpt", "gpt4", "llama-chat", "mistral-chat", "cohere-chat", "mpt-chat"]
RAID_PARAPHRASE_MODELS = ["chatgpt", "gpt4", "llama-chat", "mistral-chat"]

# Second study ("long set"): the same documents scored at several lengths.
RUNGS = [75, 150, 300, 500]
RUNG_FILL = 0.85            # a text counts for a rung when it has at least this share of the rung's words
LONG_MAX_WORDS = 500
# RAID domains that have human documents of 300 words or more (abstracts never reach that length).
LONG_RAID_DOMAINS = ["books", "news", "poetry", "recipes", "reddit", "reviews", "wiki"]
# The last Clef model appeared in the Workers AI catalogue on 2026-10-09, so text first published
# from this date on cannot have been in any of the three models' training data.
UNSEEN_FROM = "2026-10-10"


def load_env() -> None:
    """Read KEY=VALUE lines from the repository's untracked .env without overriding the environment."""
    path = REPO / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def raw_dir() -> Path:
    """Directory for raw third-party downloads; kept outside Git."""
    load_env()
    path = Path(os.environ.get("RAW_DATA_DIR", REPO / "raw"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def cap_gpu_memory(share: str = "0.3") -> None:
    """Call before importing torch. Caps Apple-GPU memory at a share of what the system recommends, so a
    job that grows fails with an out-of-memory error instead of pushing the whole machine into swap."""
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", share)
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.2")


def torch_device() -> str:
    """Device for the local detectors: DETECTEVAL_DEVICE if set, else Apple GPU, CUDA or CPU."""
    import torch

    if os.environ.get("DETECTEVAL_DEVICE"):
        return os.environ["DETECTEVAL_DEVICE"]
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
