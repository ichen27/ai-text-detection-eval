"""Compile data/long_set.jsonl: documents of 300 words or more, to be scored at several lengths.

Compared with the first evaluation set this one
  * keeps up to 500 words per text, so the same document can be cut to 75, 150, 300 and 500 words;
  * draws RAID texts from seven domains instead of four;
  * adds human forum posts published after the Clef models were released (unseen in training),
    and posts from the same sites in 2019 as a control;
  * adds machine-written forum posts and student essays, so those genres are not human-only;
  * assigns every text to a "calib" or "test" half, so thresholds are never set on the texts they are
    reported on. Texts that share a source document stay in the same half.

Run once before ``generate_long`` (to choose the human sources) and again after it.
"""
from __future__ import annotations

import collections
import hashlib
import json
import random

import duckdb

from . import config
from .build_dataset import fill_placeholders
from .fetch_raid_full import parquet_path
from .textprep import rung_text, strip_markdown, strip_preamble, trim_words

RAID_HUMANS = {"books": 40, "news": 40, "poetry": 30, "recipes": 30, "reviews": 30, "reddit": 20, "wiki": 20}
RAID_AI_PER_MODEL_DOMAIN = 4
RAID_PARAPHRASED_PER_MODEL_DOMAIN = 8
RAID_PARAPHRASED_DOMAINS = ["books", "news"]   # the only domains whose paraphrased texts reach 300 words
FORUM_2019 = 120
_WORDS = "(length(generation) - length(replace(generation, ' ', '')) + 1)"


def split_of(key: str) -> str:
    """Stable half assignment from a source key."""
    return "calib" if int(hashlib.sha256(key.encode()).hexdigest(), 16) % 2 == 0 else "test"


def prepare(text: str) -> str | None:
    """Clean a text and keep it only if it can fill the 300-word rung."""
    text = strip_markdown(text)
    if rung_text(text, 300, config.RUNG_FILL) is None:
        return None
    return trim_words(text, config.LONG_MAX_WORDS, 1)


def record(identifier: str, label: str, group: str, text: str, split_key: str, **extra) -> dict | None:
    prepared = prepare(text)
    if prepared is None:
        return None
    return {"id": identifier, "label": label, "group": group, "split": split_of(split_key), **extra,
            "words": len(prepared.split()), "text": prepared}


def raid_records(seed: int) -> list[dict]:
    rng = random.Random(seed)
    con = duckdb.connect()
    full = str(parquet_path())
    out, chosen_sources = [], collections.defaultdict(set)
    for domain, wanted in RAID_HUMANS.items():
        rows = con.execute(f"""SELECT id, source_id, title, generation FROM read_parquet(?)
                               WHERE model = 'human' AND domain = ? AND {_WORDS} >= 290 ORDER BY id""",
                           [full, domain]).fetchall()
        rng.shuffle(rows)
        kept = 0
        for row_id, source_id, title, text in rows:
            item = record("raid-" + row_id[:13], "human", "raid_human", text, source_id, source="RAID",
                          domain=domain, generator="human", source_id=source_id, title=title)
            if item and kept < wanted:
                out.append(item)
                chosen_sources[domain].add(source_id)
                kept += 1
    for domain in RAID_HUMANS:
        for model in config.RAID_CHAT_MODELS:
            rows = con.execute(f"""SELECT id, source_id, title, generation FROM read_parquet(?)
                                   WHERE model = ? AND domain = ? AND decoding = 'sampling'
                                     AND repetition_penalty = 'no' AND {_WORDS} >= 290 ORDER BY id""",
                               [full, model, domain]).fetchall()
            rng.shuffle(rows)
            # Prefer generations written for the chosen human documents, so topics match.
            rows.sort(key=lambda row, d=domain: row[1] not in chosen_sources[d])
            kept = 0
            for row_id, source_id, title, text in rows:
                item = record("raid-" + row_id[:13], "ai", "raid_2023", text, source_id, source="RAID",
                              domain=domain, generator=model, source_id=source_id, title=title)
                if item and kept < RAID_AI_PER_MODEL_DOMAIN:
                    out.append(item)
                    kept += 1
    attacked = str(config.raw_dir() / "raid" / "parquet" / "*.parquet")
    for domain in RAID_PARAPHRASED_DOMAINS:
        for model in config.RAID_PARAPHRASE_MODELS:
            rows = con.execute(f"""SELECT id, source_id, title, generation FROM read_parquet(?)
                                   WHERE model = ? AND domain = ? AND attack = 'paraphrase'
                                     AND decoding = 'sampling' AND repetition_penalty = 'no'
                                     AND {_WORDS} >= 290 ORDER BY id""", [attacked, model, domain]).fetchall()
            rng.shuffle(rows)
            kept = 0
            for row_id, source_id, title, text in rows:
                item = record("raid-" + row_id[:13], "ai", "raid_paraphrased", text, source_id, source="RAID",
                              domain=domain, generator=model, source_id=source_id, title=title)
                if item and kept < RAID_PARAPHRASED_PER_MODEL_DOMAIN:
                    out.append(item)
                    kept += 1
    return out


def essay_records() -> list[dict]:
    out = []
    with open(config.raw_dir() / "liang2023" / "HewlettStudentEssay_real_88.json") as handle:
        for index, essay in enumerate(json.load(handle)):
            identifier = f"essay_native-{index:03d}"
            item = record(identifier, "human", "essay_native", fill_placeholders(essay["document"]), identifier,
                          source="Liang et al. 2023", domain="essay", generator="human", source_id=identifier,
                          title="")
            if item:
                out.append(item)
    return out


def forum_records(seed: int) -> list[dict]:
    """Recent posts are all kept; 2019 posts are sampled to match the recent sites where possible."""
    directory = config.raw_dir() / "stackexchange"

    def load(pattern: str) -> list[dict]:
        rows = []
        for path in sorted(directory.glob(pattern)):
            with open(path) as handle:
                rows += [json.loads(line) for line in handle]
        return sorted(rows, key=lambda row: (row["site"], row["post_id"]))

    def to_item(row: dict, group: str) -> dict | None:
        identifier = f"{group}-{row['site']}-{row['post_id']}"
        return record(identifier, "human", group, row["text"], identifier, source="Stack Exchange (CC BY-SA 4.0)",
                      domain="forum", generator="human", source_id=identifier, title=row["title"],
                      site=row["site"], link=row["link"], author=row["author"], created=row["created"])

    recent = [item for row in load("eval_recent*.jsonl") if (item := to_item(row, "forum_recent"))]
    per_site = collections.Counter(item["site"] for item in recent)
    rng = random.Random(seed + 2)
    older = load("eval_2019*.jsonl")
    rng.shuffle(older)
    matched, spare, taken = [], [], collections.Counter()
    for row in older:
        item = to_item(row, "forum_2019")
        if item is None:
            continue
        if taken[row["site"]] < per_site[row["site"]]:
            matched.append(item)
            taken[row["site"]] += 1
        else:
            spare.append(item)
    return recent + (matched + spare)[:FORUM_2019]


def generated_records() -> list[dict]:
    path = config.DATA_DIR / "generated_long.jsonl"
    if not path.exists():
        return []
    groups = {"fresh": ("ai", "fresh_2026"), "humanized": ("ai", "humanized_2026"),
              "polished": ("mixed", "polished_2026")}
    out = []
    with open(path) as handle:
        for row in map(json.loads, handle):
            label, group = groups[row["kind"]]
            extra = {"rewriter": row["rewriter"].rsplit("/", 1)[1]} if row["kind"] != "fresh" else {}
            generator = "human" if row["kind"] == "polished" else row["generator"].rsplit("/", 1)[1]
            item = record(row["id"], label, group, strip_preamble(row["text"]), row["source_id"],
                          source="generated for this study", domain=row["domain"], generator=generator,
                          source_id=row["source_id"], title=row.get("title", ""), **extra)
            if item:
                out.append(item)
    return out


def load_long_set() -> list[dict]:
    with open(config.DATA_DIR / "long_set.jsonl") as handle:
        return [json.loads(line) for line in handle]


def main() -> None:
    records = raid_records(config.SEED) + essay_records() + forum_records(config.SEED) + generated_records()
    ids = [item["id"] for item in records]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate ids in the long set")
    records.sort(key=lambda item: item["id"])
    with open(config.DATA_DIR / "long_set.jsonl", "w") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = collections.defaultdict(list)
    for item in records:
        summary[(item["label"], item["group"])].append(item)
    for (label, group), items in sorted(summary.items()):
        words = sorted(item["words"] for item in items)
        calib = sum(item["split"] == "calib" for item in items)
        full = sum(rung_text(item["text"], 500, config.RUNG_FILL) is not None for item in items)
        print(f"{label:6} {group:18} n={len(items):4} calib={calib:3} median words={words[len(words) // 2]:3} "
              f"fill 500-word rung={full}")
    print("total", len(records))


if __name__ == "__main__":
    main()
