"""Normalise and trim texts so every group is compared at a similar length and layout."""
from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"[.!?][\"')\]]?$")
_MARKDOWN = re.compile(r"(\*\*|__|^#{1,6}\s+|^\s*[-*]\s+|`)", re.MULTILINE)


def strip_markdown(text: str) -> str:
    """Remove chat-style markdown so formatting alone cannot reveal a generated text."""
    return _MARKDOWN.sub("", text)


_PREAMBLE = re.compile(r"\A\s*(here|sure|certainly|okay|of course)\b[^\n]*:\s*\n", re.IGNORECASE)


def strip_preamble(text: str) -> str:
    """Drop a leading chat line such as "Here is the rewritten text:"."""
    return _PREAMBLE.sub("", text)


def normalise(text: str) -> str:
    """Collapse all whitespace to single spaces."""
    return " ".join(text.split())


def trim_words(text: str, max_words: int = 150, min_words: int = 60) -> str | None:
    """Return the first ``max_words`` words, cut back to a sentence end when one leaves at least
    ``min_words`` words. Returns None when the text is shorter than ``min_words``."""
    if min_words < 1 or max_words < min_words:
        raise ValueError("require 1 <= min_words <= max_words")
    words = normalise(text).split(" ")
    if len(words) < min_words or words == [""]:
        return None
    if len(words) <= max_words:
        return " ".join(words)
    window = words[:max_words]
    for end in range(len(window), min_words - 1, -1):
        if _SENTENCE_END.search(window[end - 1]):
            return " ".join(window[:end])
    return " ".join(window)
