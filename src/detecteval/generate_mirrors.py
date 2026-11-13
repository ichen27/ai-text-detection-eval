"""Generate training "mirrors": current-model texts on the titles of human training documents.

A classifier trained only on RAID has seen nothing newer than 2023 models. Following the idea behind
Pangram's synthetic mirrors (Emi and Spero, 2024), each sampled human training document gets a machine
counterpart on the same title and genre, written by one of four current models. The other four
current models are never used for training, so the evaluation can separate "generators seen in
training" from "generators never seen".

    python -m detecteval.generate_mirrors --shard 0 --of 3     # run the shards in parallel
    python -m detecteval.generate_mirrors --merge

Shards append to data/mirrors/shard-<n>.jsonl; --merge writes data/train_mirrors.jsonl.
"""
from __future__ import annotations

import argparse
import json
import random

import duckdb

from . import config
from .fetch_raid_full import parquet_path
from .generate import run_jobs, short_name
from .generate_long import BATCH, LENGTH_NOTE, MAX_TOKENS, NO_REASONING, PROMPTS
from .hub import WorkersAI

TRAIN_GENERATORS = ["@cf/openai/gpt-oss-120b", "@cf/mistralai/mistral-small-3.1-24b-instruct",
                    "@cf/google/gemma-4-26b-a4b-it", "@cf/zai-org/glm-5.3"]
RAID_TITLES_PER_DOMAIN = 75
FORUM_TITLES = 400
WORD_TARGETS = [150, 300, 450]
ABSTRACT_PROMPT = 'Write the abstract for the academic paper titled "{title}".'


def eval_source_ids() -> set[str]:
    """Source documents used by either evaluation set; never used for training."""
    ids = set()
    for name in ("eval_set.jsonl", "long_set.jsonl"):
        with open(config.DATA_DIR / name) as handle:
            ids |= {row.get("source_id", row["id"]) for row in map(json.loads, handle)}
    return ids


def jobs(seed: int) -> list[dict]:
    rng = random.Random(seed + 20)
    excluded = eval_source_ids()
    con = duckdb.connect()
    sources = []
    for domain in sorted({*config.LONG_RAID_DOMAINS, "abstracts"}):
        rows = con.execute("""SELECT source_id, title FROM read_parquet(?) WHERE model = 'human' AND domain = ?
                              AND title IS NOT NULL ORDER BY source_id""", [str(parquet_path()), domain]).fetchall()
        rows = [row for row in rows if row[0] not in excluded and row[1].strip()]
        for source_id, title in rng.sample(rows, RAID_TITLES_PER_DOMAIN):
            prompt = (ABSTRACT_PROMPT if domain == "abstracts" else PROMPTS[domain]).format(title=title.strip())
            sources.append({"domain": domain, "source_id": source_id, "title": title.strip(), "prompt": prompt})
    with open(config.raw_dir() / "stackexchange" / "train.jsonl") as handle:
        posts = [row for row in map(json.loads, handle) if row["type"] == "question" and row["title"].strip()]
    for post in rng.sample(sorted(posts, key=lambda row: (row["site"], row["post_id"])), FORUM_TITLES):
        sources.append({"domain": "forum", "source_id": f"forum_train-{post['site']}-{post['post_id']}",
                        "title": post["title"],
                        "prompt": PROMPTS["forum"].format(title=post["title"], site=post["site"])})
    rng.shuffle(sources)
    out = []
    for n, source in enumerate(sources):
        generator = TRAIN_GENERATORS[n % len(TRAIN_GENERATORS)]
        words = WORD_TARGETS[(n // len(TRAIN_GENERATORS)) % len(WORD_TARGETS)]
        out.append({"id": f"mirror-{short_name(generator)}-{source['source_id'][-24:]}", "kind": "mirror",
                    "generator": generator, "domain": source["domain"], "source_id": source["source_id"],
                    "title": source["title"], "words_asked": words,
                    "prompt": source["prompt"] + LENGTH_NOTE.format(words=words)})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--of", type=int, default=1)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    directory = config.DATA_DIR / "mirrors"
    if args.merge:
        rows = {}
        for path in sorted(directory.glob("shard-*.jsonl")):
            with open(path) as handle:
                rows.update((row["id"], row) for row in map(json.loads, handle))
        with open(config.DATA_DIR / "train_mirrors.jsonl", "w") as handle:
            for key in sorted(rows):
                handle.write(json.dumps(rows[key], ensure_ascii=False) + "\n")
        print("merged", len(rows))
        return
    config.load_env()
    directory.mkdir(parents=True, exist_ok=True)
    finished = set()
    for path in directory.glob("shard-*.jsonl"):
        with open(path) as handle:
            finished |= {json.loads(line)["id"] for line in handle}
    mine = [job for n, job in enumerate(jobs(config.SEED)) if n % args.of == args.shard and job["id"] not in finished]
    run_jobs(WorkersAI(), mine, directory / f"shard-{args.shard}.jsonl", "generator", MAX_TOKENS, BATCH, NO_REASONING)


if __name__ == "__main__":
    main()
