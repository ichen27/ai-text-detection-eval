"""Fetch human-written Stack Exchange questions (CC BY-SA 4.0) for three purposes.

    eval_recent   questions posted after the Clef models were released, so the models cannot have
                  seen them in training (they may still include some AI-assisted writing)
    eval_2019     questions from the same sites and the same weeks of 2019: certainly written before
                  modern language models, but possibly present in training data
    train         2015-2019 questions and answers from a disjoint list of sites, for training a classifier

    python -m detecteval.fetch_stackexchange

Uses the public API without a key (300 requests per day per address). Each batch is written to
RAW_DATA_DIR/stackexchange/<name>.jsonl and skipped when the file already exists.
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import time
import urllib.error
import urllib.parse
import urllib.request

from . import config
from .textprep import html_to_text, normalise

API = "https://api.stackexchange.com/2.3/{kind}?"
EVAL_SITES = ["workplace", "academia", "interpersonal", "parenting", "travel", "money", "law", "writing",
              "worldbuilding", "rpg", "diy", "philosophy", "politics", "christianity", "softwareengineering"]
# Stack Exchange is quiet in 2026, so a second list of sites tops up the recent sample.
EVAL_SITES_EXTRA = ["electronics", "aviation", "space", "hermeneutics", "judaism", "islam", "buddhism", "hinduism",
                    "puzzling", "gaming", "anime", "chess", "linguistics", "psychology", "sports", "woodworking",
                    "genealogy", "homebrew", "crafts", "hsm", "sound", "avp", "freelancing", "pm", "ux",
                    "opensource", "retrocomputing", "ham", "martialarts", "vegetarianism"]
TRAIN_SITES = ["cooking", "pets", "outdoors", "history", "scifi", "movies", "literature", "ell", "english",
               "security", "bicycles", "gardening", "mechanics", "photo", "music", "boardgames", "skeptics",
               "lifehacks", "expatriates", "fitness"]
MIN_WORDS_EVAL = 300
MIN_WORDS_TRAIN = 80


def epoch(date: str) -> int:
    return int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.UTC).timestamp())


def get(kind: str, **params) -> dict:
    """One API page. Honours the server's ``backoff`` request and retries throttling once."""
    url = API.format(kind=kind) + urllib.parse.urlencode({"pagesize": 100, "filter": "withbody", **params})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"Accept-Encoding": "gzip"}),
                                        timeout=60) as response:
                raw = response.read()
                if response.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
            page = json.loads(raw)
            time.sleep(page.get("backoff", 0) + 0.3)
            return page
        except urllib.error.HTTPError as error:
            if error.code not in (429, 502, 503) or attempt == 2:
                raise
            time.sleep(30)
    raise RuntimeError("unreachable")


def to_record(site: str, kind: str, item: dict) -> dict:
    text = normalise(html_to_text(item["body"]))
    post_id = item.get("question_id") if kind == "questions" else item.get("answer_id")
    return {
        "site": site, "type": kind[:-1], "post_id": post_id, "title": html_to_text(item.get("title", "")),
        "link": item.get("link") or f"https://{site}.stackexchange.com/a/{post_id}",
        "author": html_to_text(item.get("owner", {}).get("display_name", "anonymous")),
        "created": dt.datetime.fromtimestamp(item["creation_date"], dt.UTC).date().isoformat(),
        "score": item.get("score", 0), "closed": "closed_date" in item,
        "words": len(text.split()), "text": text,
    }


def fetch(site: str, kind: str, start: str, end: str, pages: int, min_words: int) -> list[dict]:
    out = []
    for page_number in range(1, pages + 1):
        page = get(kind, site=site, fromdate=epoch(start), todate=epoch(end), order="desc", sort="creation",
                   page=page_number)
        for item in page.get("items", []):
            record = to_record(site, kind, item)
            if record["words"] >= min_words and record["score"] >= 0 and not record["closed"]:
                out.append(record)
        if not page.get("has_more"):
            break
    return out


def batch(name: str, jobs: list[tuple]) -> None:
    directory = config.raw_dir() / "stackexchange"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.jsonl"
    if path.exists():
        print(name, "already fetched")
        return
    records = []
    for site, kind, start, end, pages, min_words in jobs:
        try:
            found = fetch(site, kind, start, end, pages, min_words)
        except urllib.error.HTTPError as error:
            print(f"  {site} {kind}: HTTP {error.code}, skipped")
            continue
        print(f"  {site} {kind} {start}..{end}: {len(found)}", flush=True)
        records += found
    with open(path, "w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(name, len(records))


def main() -> None:
    today = dt.datetime.now(dt.UTC).date().isoformat()
    batch("eval_recent", [(site, "questions", config.UNSEEN_FROM, today, 3, MIN_WORDS_EVAL) for site in EVAL_SITES])
    batch("eval_2019", [(site, "questions", "2019-10-10", "2019-11-13", 2, MIN_WORDS_EVAL) for site in EVAL_SITES])
    batch("eval_recent_extra",
          [(site, "questions", config.UNSEEN_FROM, today, 1, MIN_WORDS_EVAL) for site in EVAL_SITES_EXTRA])
    batch("eval_2019_extra",
          [(site, "questions", "2019-10-10", "2019-11-13", 1, MIN_WORDS_EVAL) for site in EVAL_SITES_EXTRA])
    train = [(site, "questions", "2017-03-01", "2017-06-01", 1, MIN_WORDS_TRAIN) for site in TRAIN_SITES]
    train += [(site, "answers", "2018-03-01", "2018-06-01", 2, MIN_WORDS_TRAIN) for site in TRAIN_SITES]
    batch("train", train)


if __name__ == "__main__":
    main()
