"""Compile data/eval_set.jsonl from the RAID sample, the Liang et al. essays and data/generated.jsonl.

Every text gets the same treatment: chat markdown removed, whitespace collapsed, and trimmed to at most
MAX_WORDS words at a sentence end (texts under MIN_WORDS are dropped). Labels are "human", "ai" or
"mixed" (human text polished by GPT-4, which has no clean label and is reported separately).
"""
from __future__ import annotations

import collections
import json
import random
import re

from . import config
from .textprep import strip_markdown, strip_preamble, trim_words

RAID_HUMANS_PER_DOMAIN = 40

# The native-speaker essays (ASAP/Hewlett) were anonymised with tokens such as @CAPS1 or @LOCATION2.
# Left in place they mark a text as coming from a human corpus, so they are replaced with plain fillers.
_PLACEHOLDER = re.compile(r"@([A-Z]+)\d*")
_FILLERS = {"PERSON": "Alex", "LOCATION": "Ohio", "CITY": "Dayton", "STATE": "Ohio", "ORGANIZATION": "Google",
            "CAPS": "Local", "NUM": "ten", "PERCENT": "50%", "DATE": "June 1", "TIME": "3:00", "MONTH": "June",
            "MONEY": "$20", "DR": "Dr. Lee", "EMAIL": "me@example.com"}


def fill_placeholders(text: str) -> str:
    return _PLACEHOLDER.sub(lambda match: _FILLERS.get(match.group(1), "it"), text)


def prepare(text: str) -> str | None:
    return trim_words(strip_markdown(text), config.MAX_WORDS, config.MIN_WORDS)


def record(identifier: str, label: str, group: str, text: str, **extra) -> dict | None:
    prepared = prepare(text)
    if prepared is None:
        return None
    return {"id": identifier, "label": label, "group": group, **extra,
            "words": len(prepared.split()), "text": prepared}


def raid_records(seed: int) -> list[dict]:
    with open(config.raw_dir() / "raid" / "raid_sample.jsonl") as handle:
        rows = [json.loads(line) for line in handle]
    rng = random.Random(seed)
    out, humans_kept = [], collections.Counter()
    humans = sorted((row for row in rows if row["model"] == "human"), key=lambda row: row["id"])
    rng.shuffle(humans)
    for row in humans:
        if humans_kept[row["domain"]] >= RAID_HUMANS_PER_DOMAIN:
            continue
        item = record("raid-" + row["id"][:13], "human", "raid_human", row["generation"],
                      source="RAID", domain=row["domain"], generator="human", source_id=row["source_id"])
        if item:
            humans_kept[row["domain"]] += 1
            out.append(item)
    for row in rows:
        if row["model"] == "human":
            continue
        group = "raid_paraphrased" if row["attack"] == "paraphrase" else "raid_2023"
        item = record("raid-" + row["id"][:13], "ai", group, row["generation"], source="RAID",
                      domain=row["domain"], generator=row["model"], source_id=row["source_id"])
        if item:
            out.append(item)
    return out


def essay_records() -> list[dict]:
    sets = [("TOEFL_real_91", "human", "essay_nonnative", "human"),
            ("HewlettStudentEssay_real_88", "human", "essay_native", "human"),
            ("TOEFL_gpt4polished_91", "mixed", "toefl_gpt4_polished", "human+gpt4")]
    out = []
    for name, label, group, generator in sets:
        with open(config.raw_dir() / "liang2023" / f"{name}.json") as handle:
            for index, essay in enumerate(json.load(handle)):
                document = fill_placeholders(essay["document"]) if group == "essay_native" else essay["document"]
                item = record(f"{group}-{index:03d}", label, group, document,
                              source="Liang et al. 2023", domain="essay", generator=generator)
                if item:
                    out.append(item)
    return out


def generated_records() -> list[dict]:
    out = []
    with open(config.DATA_DIR / "generated.jsonl") as handle:
        for row in map(json.loads, handle):
            group = "fresh_2026" if row["kind"] == "fresh" else "humanized_2026"
            generator = row["generator"].rsplit("/", 1)[1]
            extra = {"rewriter": row["rewriter"].rsplit("/", 1)[1]} if row["kind"] == "humanized" else {}
            item = record(row["id"], "ai", group, strip_preamble(row["text"]), source="generated for this study",
                          domain=row["domain"], generator=generator, source_id=row["source_id"], **extra)
            if item:
                out.append(item)
    return out


def main() -> None:
    records = raid_records(config.SEED) + essay_records() + generated_records()
    ids = [item["id"] for item in records]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate ids in the evaluation set")
    records.sort(key=lambda item: item["id"])
    with open(config.DATA_DIR / "eval_set.jsonl", "w") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = collections.defaultdict(list)
    for item in records:
        summary[(item["label"], item["group"])].append(item["words"])
    for (label, group), words in sorted(summary.items()):
        words.sort()
        print(f"{label:6} {group:22} n={len(words):4} median words={words[len(words) // 2]}")
    print("total", len(records))


if __name__ == "__main__":
    main()
