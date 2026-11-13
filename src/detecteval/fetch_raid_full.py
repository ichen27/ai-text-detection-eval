"""Download RAID's full non-adversarial training split (eight domains, about 468,000 rows, 0.8 GB).

The Hugging Face parquet conversion used by ``fetch_raid`` covers four domains and includes the
adversarial attacks. This file from the benchmark's own host adds recipes, reddit, reviews and wiki.

    python -m detecteval.fetch_raid_full
"""
from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

import duckdb

from . import config

URL = "https://dataset.raid-bench.xyz/train_none.csv"


def parquet_path() -> Path:
    return config.raw_dir() / "raid" / "full" / "train_none.parquet"


def main() -> None:
    target = parquet_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print("already present:", target)
        return
    csv = target.with_suffix(".csv")
    if not csv.exists():
        with urllib.request.urlopen(URL, timeout=600) as response, open(csv.with_suffix(".part"), "wb") as out:
            shutil.copyfileobj(response, out, 1 << 20)
        csv.with_suffix(".part").rename(csv)
    con = duckdb.connect()
    con.execute(f"""COPY (SELECT * FROM read_csv(?, header=true, quote='"', escape='"',
                    max_line_size=10000000, all_varchar=true)) TO '{target}' (FORMAT parquet)""", [str(csv)])
    print("rows:", con.execute("SELECT count(*) FROM read_parquet(?)", [str(target)]).fetchone()[0])


if __name__ == "__main__":
    main()
