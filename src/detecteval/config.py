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
