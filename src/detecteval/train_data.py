"""Assemble training examples for the classifier. Nothing here overlaps either evaluation set.

Two recipes:
  raid      RAID human documents and 2023-era machine text only (eight domains, eleven generators)
  mirrors   an update set for a model already trained on ``raid``: "mirror" texts written by four
            current models on the titles of human training documents (see ``generate_mirrors``),
            human Stack Exchange posts from 2017-2018, and a sample of the RAID examples so the
            model does not forget them

Exclusions that keep the evaluation honest:
  * every RAID source document used by an evaluation set, with all machine text generated from it;
  * the Stack Exchange sites used for evaluation (training posts come from a disjoint list of sites);
  * four of the eight current generators, which appear only in the evaluation sets;
  * student essays, which appear only in the evaluation sets.
"""
from __future__ import annotations

import json
import random

import duckdb

from . import config
from .fetch_raid_full import parquet_path
from .generate_mirrors import eval_source_ids
from .textprep import normalise, strip_markdown, strip_preamble

AI_PER_HUMAN = 2
MIRROR_REPEATS = 3          # mirrors are few; each is seen this many times per epoch, cropped differently
RAID_REPLAY = 4000          # RAID examples mixed into the mirror update
VALIDATION_SHARE = 0.04
MIN_WORDS = 50


def clean(text: str) -> str:
    return normalise(strip_markdown(text))


def raid_examples(seed: int) -> list[dict]:
    rng = random.Random(seed + 30)
    excluded = eval_source_ids()
    con = duckdb.connect()
    path = str(parquet_path())
    humans = con.execute("""SELECT source_id, domain, generation FROM read_parquet(?) WHERE model = 'human'
                            ORDER BY source_id""", [path]).fetchall()
    humans = [row for row in humans if row[0] not in excluded]
    out = [{"text": clean(text), "label": 0, "domain": domain, "generator": "human", "source_id": source_id}
           for source_id, domain, text in humans]
    index = con.execute("""SELECT id, source_id, model FROM read_parquet(?) WHERE model <> 'human'
                           ORDER BY id""", [path]).fetchall()
    by_source: dict[str, dict[str, list[str]]] = {}
    for row_id, source_id, model in index:
        by_source.setdefault(source_id, {}).setdefault(model, []).append(row_id)
    chosen = []
    for source_id, _, _ in humans:
        models = by_source.get(source_id, {})
        for model in rng.sample(sorted(models), min(AI_PER_HUMAN, len(models))):
            chosen.append(rng.choice(models[model]))
    rows = con.execute("""SELECT r.source_id, r.domain, r.model, r.generation FROM read_parquet(?) r
                          WHERE r.id IN (SELECT unnest(?)) ORDER BY r.id""", [path, chosen]).fetchall()
    out += [{"text": clean(text), "label": 1, "domain": domain, "generator": model, "source_id": source_id}
            for source_id, domain, model, text in rows]
    return out


def mirror_examples() -> list[dict]:
    out = []
    with open(config.raw_dir() / "stackexchange" / "train.jsonl") as handle:
        for row in map(json.loads, handle):
            out.append({"text": row["text"], "label": 0, "domain": "forum", "generator": "human",
                        "source_id": f"forum_train-{row['site']}-{row['post_id']}"})
    with open(config.DATA_DIR / "train_mirrors.jsonl") as handle:
        for row in map(json.loads, handle):
            example = {"text": clean(strip_preamble(row["text"])), "label": 1, "domain": row["domain"],
                       "generator": row["generator"].rsplit("/", 1)[1], "source_id": row["source_id"]}
            out += [example] * MIRROR_REPEATS
    return out


def load(recipe: str, seed: int = config.SEED) -> tuple[list[dict], list[dict]]:
    """Return (training examples, validation examples), split by source document."""
    if recipe not in ("raid", "mirrors"):
        raise ValueError("recipe must be 'raid' or 'mirrors'")
    examples = raid_examples(seed)
    if recipe == "mirrors":
        examples = random.Random(seed + 32).sample(examples, RAID_REPLAY) + mirror_examples()
    examples = [example for example in examples if len(example["text"].split()) >= MIN_WORDS]
    sources = sorted({example["source_id"] for example in examples})
    random.Random(seed + 31).shuffle(sources)
    held_out = set(sources[: int(len(sources) * VALIDATION_SHARE)])
    train = [example for example in examples if example["source_id"] not in held_out]
    validation = [example for example in examples if example["source_id"] in held_out]
    return train, validation
