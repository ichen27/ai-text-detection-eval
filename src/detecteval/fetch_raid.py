"""Draw a seeded, topic-paired sample from the RAID benchmark (Dugan et al., ACL 2024; MIT licence).

Hugging Face hosts a partial parquet conversion of RAID's training split covering four domains
(about 2.3 million rows). This module downloads those files, picks human documents per domain, then
picks machine generations for the same source documents so that human and AI texts share topics.
"""
from __future__ import annotations

import json
import random
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

import duckdb

from . import config

PARQUET_API = "https://huggingface.co/api/datasets/liamdugan/raid/parquet/raid/train/{}.parquet"
HUMANS_PER_DOMAIN = 60          # oversampled; the dataset builder keeps those long enough
AI_PER_MODEL_DOMAIN = 9
PARAPHRASE_PER_MODEL_DOMAIN = 5
PARQUET_FILES = 10


def download(index: int, target: Path) -> None:
    """Download one parquet file in a single request, waiting out Hugging Face rate limits."""
    partial = target.with_suffix(".part")
    for _ in range(10):
        try:
            with urllib.request.urlopen(PARQUET_API.format(index), timeout=120) as response, open(partial, "wb") as out:
                shutil.copyfileobj(response, out, 1 << 20)
            partial.rename(target)
            return
        except urllib.error.HTTPError as error:
            if error.code != 429:
                raise
            time.sleep(int(error.headers.get("retry-after", "30")) + 2)
    raise RuntimeError("Hugging Face rate limit did not clear")


def download_all(directory: Path, count: int = PARQUET_FILES) -> list[str]:
    """Whole-file downloads use one request each; remote range reads exhaust the anonymous rate limit."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        target = directory / f"{index:04d}.parquet"
        if not target.exists():
            download(index, target)
            print("downloaded", target.name, f"{target.stat().st_size / 1e6:.0f} MB", flush=True)
        paths.append(str(target))
    return paths


def build_index(con: duckdb.DuckDBPyConnection, paths: list[str]) -> None:
    con.execute("""
        CREATE OR REPLACE TEMP TABLE idx AS
        SELECT filename AS url, file_row_number AS r, model, attack, domain, decoding, repetition_penalty, source_id
        FROM read_parquet(?, filename=true, file_row_number=true)
    """, [paths])


def choose_rows(con: duckdb.DuckDBPyConnection, seed: int) -> list[tuple[str, int]]:
    rng = random.Random(seed)
    chosen: list[tuple[str, int]] = []
    for domain in config.RAID_DOMAINS:
        humans = con.execute(
            "SELECT url, r, source_id FROM idx WHERE model='human' AND attack='none' AND domain=? ORDER BY source_id",
            [domain]).fetchall()
        humans = rng.sample(humans, HUMANS_PER_DOMAIN)
        chosen += [(url, r) for url, r, _ in humans]
        sources = [source for _, _, source in humans]
        plans = [(model, "none", AI_PER_MODEL_DOMAIN) for model in config.RAID_CHAT_MODELS]
        plans += [(model, "paraphrase", PARAPHRASE_PER_MODEL_DOMAIN) for model in config.RAID_PARAPHRASE_MODELS]
        for model, attack, wanted in plans:
            # One generation per source document, preferring sampling without repetition penalty.
            rows = con.execute("""
                SELECT url, r, source_id FROM (
                  SELECT *, row_number() OVER (PARTITION BY source_id
                    ORDER BY (repetition_penalty = 'no') DESC, (decoding = 'sampling') DESC, r) AS pick
                  FROM idx WHERE model=? AND attack=? AND domain=? AND source_id IN (SELECT unnest(?))
                ) WHERE pick = 1 ORDER BY source_id
            """, [model, attack, domain, sources]).fetchall()
            chosen += [(url, r) for url, r, _ in rng.sample(rows, min(wanted, len(rows)))]
    return chosen


def fetch_rows(con: duckdb.DuckDBPyConnection, chosen: list[tuple[str, int]]) -> list[dict]:
    by_url: dict[str, list[int]] = {}
    for url, r in chosen:
        by_url.setdefault(url, []).append(r)
    records = []
    for url, rows in sorted(by_url.items()):
        cursor = con.execute("""
            SELECT id, source_id, model, attack, domain, decoding, repetition_penalty, title, generation
            FROM read_parquet(?, file_row_number=true) WHERE file_row_number IN (SELECT unnest(?))
        """, [url, rows])
        names = [column[0] for column in cursor.description]
        records += [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]
    return records


def main() -> None:
    out_dir = config.raw_dir() / "raid"
    paths = download_all(out_dir / "parquet")
    con = duckdb.connect()
    build_index(con, paths)
    chosen = choose_rows(con, config.SEED)
    records = fetch_rows(con, chosen)
    with open(out_dir / "raid_sample.jsonl", "w") as handle:
        for record in sorted(records, key=lambda item: item["id"]):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"chose {len(chosen)} rows, wrote {len(records)} -> {out_dir / 'raid_sample.jsonl'}")


if __name__ == "__main__":
    main()
