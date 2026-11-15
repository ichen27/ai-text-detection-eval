"""Turn a text into passages, and raw detector scores into a cautious, readable result.

This module holds the logic shared by the web app and the MCP server and needs neither PyTorch nor a
web framework. A result never says "AI" or "human". For each passage and detector it reports how
often human reference text scores at least as high, and how often AI reference text scores at most as
high, and sorts that into a band by the stricter human reference available.
"""
from __future__ import annotations

from . import config
from .calibration import share_at_least, share_at_most
from .textprep import normalise, rung_text, strip_markdown, trim_words

MAX_PASSAGES = 12
MIN_WORDS = int(config.RUNGS[0] * config.RUNG_FILL)
# Bands by the share of human reference texts that score at least this high.
BANDS = [(0.01, "strong", "1% or fewer of human reference texts score this high"),
         (0.05, "moderate", "5% or fewer of human reference texts score this high"),
         (0.20, "weak", "20% or fewer of human reference texts score this high"),
         (1.01, "none", "human reference texts often score this high")]
LIMITS = [
    "This is a score with error rates, not proof. Do not use it on its own to accuse anyone.",
    "Formal school essays are flagged more often than other human writing; the private reference, when "
    "present, comes from one student's pre-ChatGPT work.",
    "Text a person wrote and an AI then polished usually scores as AI.",
    "Text from AI models newer than the training data may score as human.",
    f"Under about 150 words the result is unreliable; under {MIN_WORDS} words no result is given.",
]


def split_passages(text: str) -> tuple[list[dict], bool]:
    """Consecutive passages of up to 500 words, each cut to the longest reference length it can fill.

    Returns (passages, truncated). A passage is {"index", "words", "rung", "text"}; ``text`` is what the
    detectors score. Texts too short for the shortest reference length give no passages."""
    words = normalise(strip_markdown(text)).split(" ")
    passages, start = [], 0
    while len(words) - start >= MIN_WORDS and len(passages) < MAX_PASSAGES:
        piece = trim_words(" ".join(words[start:]), config.LONG_MAX_WORDS, MIN_WORDS)
        rung = max(r for r in config.RUNGS if rung_text(piece, r, config.RUNG_FILL) is not None)
        passages.append({"index": len(passages) + 1, "words": len(piece.split(" ")), "rung": rung,
                         "text": rung_text(piece, rung, config.RUNG_FILL)})
        start += len(piece.split(" "))
    return passages, len(words) - start >= MIN_WORDS


def band_for(human_share: float) -> tuple[str, str]:
    for limit, name, meaning in BANDS:
        if human_share <= limit:
            return name, meaning
    raise ValueError("share must be at most 1")


def interpret(detector: str, rung: int, score: float, calibration: dict) -> dict:
    """One detector's reading of one passage against the public and, if present, private references."""
    public = calibration["public"][detector][str(rung)]
    human = {"public": share_at_least(public["human"], score)}
    counts = {"public": len(public["human"])}
    private = (calibration.get("private") or {}).get(detector, {}).get(str(rung), {}).get("human")
    if private:
        human["private"] = share_at_least(private, score)
        counts["private"] = len(private)
    band, meaning = band_for(max(human.values()))
    return {"score": round(score, 4), "band": band, "meaning": meaning,
            "human_texts_scoring_at_least_this": human, "human_reference_sizes": counts,
            "ai_texts_scoring_at_most_this": share_at_most(public["ai"], score),
            "ai_reference_size": len(public["ai"])}


def summarise(passages: list[dict]) -> dict:
    """Counts of passages per band for the main detector, and a one-line plain reading."""
    bands = [passage["detectors"]["classifier-mirrors"]["band"] for passage in passages]
    counts = {name: bands.count(name) for _, name, _ in BANDS}
    total = len(bands)
    if counts["strong"] + counts["moderate"] == 0:
        reading = "No passage scores in a range that is unusual for human writing."
    elif counts["strong"] == total:
        reading = "Every passage scores in a range that 1% or fewer of human reference texts reach."
    else:
        unusual = counts["strong"] + counts["moderate"]
        reading = (f"{unusual} of {total} passage{'s' if total != 1 else ''} score in a range that 5% or fewer "
                   "of human reference texts reach.")
    return {"passages": total, "bands": counts, "reading": reading}


def build_result(passages: list[dict], truncated: bool, scores: dict[str, list[float] | None],
                 calibration: dict, notes: list[str] | None = None) -> dict:
    """Assemble the full response from passages and per-detector score lists (None = detector not run)."""
    out_passages = []
    for position, passage in enumerate(passages):
        detectors = {name: interpret(name, passage["rung"], values[position], calibration)
                     for name, values in scores.items() if values is not None}
        out_passages.append({"index": passage["index"], "words": passage["words"],
                             "scored_at_words": passage["rung"], "start": passage["text"][:90],
                             "detectors": detectors})
    notes = list(notes or [])
    if truncated:
        notes.append(f"Only the first {MAX_PASSAGES} passages were scored.")
    if any(passage["rung"] < 150 for passage in passages):
        notes.append("Some passages are under 150 words, where results are unreliable.")
    return {"ok": True, "summary": summarise(out_passages), "passages": out_passages, "notes": notes,
            "private_reference": bool(calibration.get("private")), "limits": LIMITS}


def too_short() -> dict:
    return {"ok": False, "error": f"Need at least {MIN_WORDS} words to give any result.", "limits": LIMITS}
