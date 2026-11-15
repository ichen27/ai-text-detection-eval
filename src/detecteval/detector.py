"""Turn a text into passages, and raw detector scores into a cautious, readable result.

This module holds the logic shared by the web app and the MCP server and needs neither PyTorch nor a
web framework. A result describes the score, not the author: a label from "Human-like" to "Strongly
AI-like", and one sentence saying how often human writing scores that high ("about 1 in 30"). The
label follows the stricter human reference available. The exact rates stay in the result for anyone
who wants them.
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
LABELS = {"none": "Human-like", "weak": "Slightly AI-like", "moderate": "AI-like", "strong": "Strongly AI-like"}
_ROUND_ODDS = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200, 250, 300, 400, 500, 1000]
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


def one_in(share: float) -> int:
    """A share as round odds: 0.034 -> 30, read "about 1 in 30"."""
    if not 0 < share <= 1:
        raise ValueError("share must be above 0 and at most 1")
    return min(_ROUND_ODDS, key=lambda n: abs(n - 1 / share) / n)


def plain_reading(band: str, human_share: float, human_count: int, ai_share: float | None) -> str:
    """One or two sentences a non-specialist can act on."""
    if human_share == 0:
        return f"None of the {human_count} human reference texts scored this high."
    if band != "none":
        prefix = "Only about" if band in ("moderate", "strong") else "About"
        return f"{prefix} 1 in {one_in(human_share)} human texts score this high."
    common = ("More than half of human texts score this high." if human_share >= 0.5
              else f"About 1 in {one_in(human_share)} human texts score this high, which is common.")
    if not ai_share:
        return common + " No AI reference text scored this low."
    if ai_share < 0.05:
        return common + f" Few AI texts score this low (about 1 in {one_in(ai_share)})."
    return common + f" About 1 in {one_in(ai_share)} AI texts score this low too, so this does not rule AI out."


def interpret(detector: str, rung: int, score: float, calibration: dict) -> dict:
    """One detector's reading of one passage against the public and, if present, private references."""
    public = calibration["public"][detector][str(rung)]
    human = {"public": share_at_least(public["human"], score)}
    counts = {"public": len(public["human"])}
    private = (calibration.get("private") or {}).get(detector, {}).get(str(rung), {}).get("human")
    if private:
        human["private"] = share_at_least(private, score)
        counts["private"] = len(private)
    strictest = max(human.values())
    band, meaning = band_for(strictest)
    ai_share = share_at_most(public["ai"], score)
    return {"label": LABELS[band], "plain": plain_reading(band, strictest, sum(counts.values()), ai_share),
            "band": band, "meaning": meaning, "score": round(score, 4),
            "human_texts_scoring_at_least_this": human, "human_reference_sizes": counts,
            "ai_texts_scoring_at_most_this": ai_share, "ai_reference_size": len(public["ai"])}


def summarise(passages: list[dict]) -> dict:
    """A label and one sentence for the whole text, from the main detector's reading of each passage."""
    readings = [passage["detectors"]["classifier-mirrors"] for passage in passages]
    bands = [reading["band"] for reading in readings]
    counts = {name: bands.count(name) for _, name, _ in BANDS}
    total, ai_like = len(bands), counts["strong"] + counts["moderate"]
    if total == 1:
        label, reading = readings[0]["label"], readings[0]["plain"]
    elif ai_like == total:
        label = LABELS["strong"] if counts["strong"] == total else LABELS["moderate"]
        reading = f"All {total} passages read as AI-like."
    elif ai_like:
        verb = "reads" if ai_like == 1 else "read"
        label, reading = "Mixed", f"{ai_like} of {total} passages {verb} as AI-like."
    elif counts["weak"]:
        label = "Mostly human-like"
        verb = "leans" if counts["weak"] == 1 else "lean"
        reading = f"{counts['weak']} of {total} passages {verb} slightly AI-like; none is clearly AI-like."
    else:
        label, reading = LABELS["none"], f"All {total} passages read as human-like."
    return {"passages": total, "bands": counts, "label": label, "reading": reading}


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
