"""Turn a text into passages, and raw detector scores into a cautious, readable result.

This module holds the logic shared by the web app and the MCP server and needs neither PyTorch nor a
web framework. A result describes the score, not the author. Each passage gets an AI likelihood from
1% to 99%: the chance it is AI-written if AI and human text were equally likely beforehand, read off a
curve fitted to the reference scores. A label from "Human-like" to "Strongly AI-like" follows that
number, and one sentence says how often human writing scores as high ("about 1 in 30"), using the
stricter human reference. When a private human reference is present it shares the human weight
equally with the public one, which lowers the likelihood for writing like the private texts.
"""
from __future__ import annotations

from . import config
from .calibration import likelihood, probability_curve, share_at_least, share_at_most
from .textprep import normalise, rung_text, strip_markdown, trim_words

MAX_PASSAGES = 12
MIN_WORDS = int(config.RUNGS[0] * config.RUNG_FILL)
# Bands by AI likelihood: below 30%, 30 to 60%, 60 to 85%, and 85% or more.
BANDS = [(0.30, "none"), (0.60, "weak"), (0.85, "moderate"), (1.01, "strong")]
LABELS = {"none": "Human-like", "weak": "Slightly AI-like", "moderate": "AI-like", "strong": "Strongly AI-like"}
_ROUND_ODDS = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200, 250, 300, 400, 500, 1000]
LIMITS = [
    "This is a score with error rates, not proof. Do not use it on its own to accuse anyone.",
    "The percentage assumes AI and human text were equally likely beforehand. If most of what you check is "
    "human-written, the real chance is lower than the number shown.",
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


def band_for(ai_likelihood: float) -> str:
    for limit, name in BANDS:
        if ai_likelihood < limit:
            return name
    raise ValueError("likelihood must be at most 1")


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
        prefix = "Only about" if human_share <= 0.05 else "About"
        return f"{prefix} 1 in {one_in(human_share)} human texts score this high."
    common = ("More than half of human texts score this high." if human_share >= 0.5
              else f"About 1 in {one_in(human_share)} human texts score this high, which is common.")
    if not ai_share:
        return common + " No AI reference text scored this low."
    if ai_share < 0.05:
        return common + f" Few AI texts score this low (about 1 in {one_in(ai_share)})."
    return common + f" About 1 in {one_in(ai_share)} AI texts score this low too, so this does not rule AI out."


def curves_for(detector: str, rung: int, calibration: dict) -> tuple[list[float], list[float]]:
    """The score-to-likelihood curve for one detector and length, from every human reference available."""
    public = calibration["public"][detector][str(rung)]
    private = (calibration.get("private") or {}).get(detector, {}).get(str(rung), {}).get("human")
    return probability_curve([public["human"]] + ([private] if private else []), public["ai"])


def interpret(detector: str, rung: int, score: float, calibration: dict, curve=None) -> dict:
    """One detector's reading of one passage against the public and, if present, private references."""
    public = calibration["public"][detector][str(rung)]
    human = {"public": share_at_least(public["human"], score)}
    counts = {"public": len(public["human"])}
    private = (calibration.get("private") or {}).get(detector, {}).get(str(rung), {}).get("human")
    if private:
        human["private"] = share_at_least(private, score)
        counts["private"] = len(private)
    ai_likelihood = likelihood(curve or curves_for(detector, rung, calibration), score)
    band = band_for(ai_likelihood)
    ai_share = share_at_most(public["ai"], score)
    return {"ai_likelihood_percent": round(100 * ai_likelihood), "label": LABELS[band],
            "plain": plain_reading(band, max(human.values()), sum(counts.values()), ai_share),
            "band": band, "score": round(score, 4),
            "human_texts_scoring_at_least_this": human, "human_reference_sizes": counts,
            "ai_texts_scoring_at_most_this": ai_share, "ai_reference_size": len(public["ai"])}


def summarise(passages: list[dict]) -> dict:
    """A likelihood, a label and one sentence for the whole text, from the main detector's passage readings."""
    readings = [passage["detectors"]["classifier-mirrors"] for passage in passages]
    bands = [reading["band"] for reading in readings]
    counts = {name: bands.count(name) for _, name in BANDS}
    total, ai_like = len(bands), counts["strong"] + counts["moderate"]
    words = [passage.get("words", 1) for passage in passages]
    weighted = sum(reading["ai_likelihood_percent"] * count for reading, count in zip(readings, words, strict=True))
    percent = round(weighted / sum(words))          # longer passages count for more
    if total == 1:
        label, reading = readings[0]["label"], readings[0]["plain"]
    elif ai_like == total:
        label = LABELS["strong"] if counts["strong"] == total else LABELS["moderate"]
        reading = f"All {total} passages read as AI-like."
    elif ai_like:
        verb = "reads" if ai_like == 1 else "read"
        label, reading = "Mixed", f"{ai_like} of {total} passages {verb} as AI-like."
    elif counts["weak"]:
        verb = "leans" if counts["weak"] == 1 else "lean"
        label = "Mostly human-like"
        reading = f"{counts['weak']} of {total} passages {verb} slightly AI-like; none is clearly AI-like."
    else:
        label, reading = LABELS["none"], f"All {total} passages read as human-like."
    return {"ai_likelihood_percent": percent, "passages": total, "bands": counts, "label": label,
            "reading": reading}


def build_result(passages: list[dict], truncated: bool, scores: dict[str, list[float] | None],
                 calibration: dict, notes: list[str] | None = None) -> dict:
    """Assemble the full response from passages and per-detector score lists (None = detector not run)."""
    out_passages, curves = [], {}
    for position, passage in enumerate(passages):
        detectors = {}
        for name, values in scores.items():
            if values is None:
                continue
            key = (name, passage["rung"])
            if key not in curves:
                curves[key] = curves_for(name, passage["rung"], calibration)
            detectors[name] = interpret(name, passage["rung"], values[position], calibration, curves[key])
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
