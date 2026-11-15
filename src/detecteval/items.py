"""The texts every detector is run on, as one flat list.

A "key" names one text at one length: ``<id>@short`` for the first evaluation set (at most 150 words),
and ``<id>@<rung>`` for a long-set document cut to 75, 150, 300 or 500 words.
"""
from __future__ import annotations

import json

from . import config
from .textprep import rung_text, strip_markdown


def short_items() -> list[dict]:
    with open(config.DATA_DIR / "eval_set.jsonl") as handle:
        return [{"key": row["id"] + "@short", "id": row["id"], "set": "short", "rung": 150, "text": row["text"]}
                for row in map(json.loads, handle)]


def long_items(rungs: list[int] | None = None) -> list[dict]:
    """One item per long-set document and rung the document is long enough to fill."""
    out = []
    with open(config.DATA_DIR / "long_set.jsonl") as handle:
        for row in map(json.loads, handle):
            for rung in rungs or config.RUNGS:
                text = rung_text(row["text"], rung, config.RUNG_FILL)
                if text is not None:
                    out.append({"key": f"{row['id']}@{rung}", "id": row["id"], "set": "long", "rung": rung,
                                "text": text})
    return out


def private_dir():
    """Private texts and their scores live with the raw downloads, never in the repository."""
    return config.raw_dir() / "private"


def private_documents() -> list[dict]:
    """Rows of the private manifest: ``id``, ``group``, ``text`` and any dates."""
    path = private_dir() / "manifest.jsonl"
    if not path.exists():
        raise SystemExit(f"no private manifest at {path}; see the README section on testing your own writing")
    with open(path) as handle:
        return [json.loads(line) for line in handle]


def private_items(rungs: list[int] | None = None) -> list[dict]:
    """One item per private document and length rung it can fill, cleaned like every other text."""
    out = []
    for row in private_documents():
        for rung in rungs or config.RUNGS:
            text = rung_text(strip_markdown(row["text"]), rung, config.RUNG_FILL)
            if text is not None:
                out.append({"key": f"{row['id']}@{rung}", "id": row["id"], "set": "private", "rung": rung,
                            "text": text})
    return out


def all_items() -> list[dict]:
    return short_items() + long_items()


def done_keys(path) -> set[str]:
    if not path.exists():
        return set()
    with open(path) as handle:
        return {json.loads(line)["key"] for line in handle}
