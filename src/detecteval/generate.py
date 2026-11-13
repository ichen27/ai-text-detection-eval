"""Generate fresh AI text with current Workers AI models, and "humanized" rewrites of some of it.

RAID's machine text comes from 2023-era models. This adds text from models available in late 2026,
prompted with the titles of sampled RAID human documents so topics match, plus a simple humanizer pass:
a second model rewrites the text with instructions to sound like an ordinary person.

    python -m detecteval.generate

Writes data/generated.jsonl. Finished ids are skipped, so the run can be resumed.
"""
from __future__ import annotations

import json
import random

from . import config
from .hub import WorkersAI

GENERATORS = [
    "@cf/openai/gpt-oss-120b",
    "@cf/meta/llama-4-scout-17b-16e-instruct",
    "@cf/mistralai/mistral-small-3.1-24b-instruct",
    "@cf/google/gemma-4-26b-a4b-it",
    "@cf/deepseek-ai/deepseek-v4.1-flash-1017",
    "@cf/qwen/qwen3.8-27b",
    "@cf/zai-org/glm-5.3",
    "@cf/moonshotai/kimi-k2.6",
]
REWRITERS = ["@cf/openai/gpt-oss-120b", "@cf/meta/llama-4-scout-17b-16e-instruct"]
PER_GENERATOR_DOMAIN = 4
HUMANIZED_PER_GENERATOR = 6
MAX_TOKENS = 3000  # several generators spend tokens on hidden reasoning before answering
BATCH = 2  # the MCP request times out near a minute, and one slow model fails its whole batch

PROMPTS = {
    "abstracts": 'Write the abstract for the academic paper titled "{title}".',
    "books": 'Write the body of a plot summary for a novel titled "{title}".',
    "news": 'Write the body of a BBC news article titled "{title}".',
    "poetry": 'Write the body of a poem titled "{title}".',
}
LENGTH_NOTE = " Write about 200 words. Output only the text itself, without the title or any commentary."
HUMANIZE = ("Rewrite the following text so that it reads as if an ordinary person wrote it, not an AI. "
            "Vary the sentence lengths, prefer plain words, and let it be a little uneven, but keep the "
            "meaning, the format and roughly the same length. Output only the rewritten text.\n\n{text}")


def short_name(model: str) -> str:
    return model.rsplit("/", 1)[1]


def plan(humans: list[dict], seed: int) -> list[dict]:
    """Assign each generator its own source documents in every domain."""
    rng = random.Random(seed)
    jobs = []
    for domain in config.RAID_DOMAINS:
        sources = sorted((h for h in humans if h["domain"] == domain), key=lambda h: h["source_id"])
        rng.shuffle(sources)
        for g, generator in enumerate(GENERATORS):
            for source in sources[g * PER_GENERATOR_DOMAIN:(g + 1) * PER_GENERATOR_DOMAIN]:
                jobs.append({
                    "id": f"fresh-{short_name(generator)}-{source['source_id'][:8]}", "kind": "fresh",
                    "generator": generator, "domain": domain, "source_id": source["source_id"],
                    "title": source["title"].strip(),
                    "prompt": PROMPTS[domain].format(title=source["title"].strip()) + LENGTH_NOTE,
                })
    return jobs


def extract_text(result: dict) -> str:
    choices = result.get("choices") or []
    content = choices[0].get("message", {}).get("content") if choices else result.get("response")
    return (content or "").strip()


def run_jobs(workers: WorkersAI, jobs: list[dict], path, model_key: str) -> None:
    done = set()
    if path.exists():
        with open(path) as handle:
            done = {json.loads(line)["id"] for line in handle}
    todo = [job for job in jobs if job["id"] not in done]
    with open(path, "a") as handle:
        for start in range(0, len(todo), BATCH):
            chunk = todo[start:start + BATCH]
            items = [{"model": job[model_key], "body": {
                "messages": [{"role": "user", "content": job["prompt"]}],
                "max_tokens": MAX_TOKENS, "temperature": 1.0}} for job in chunk]
            for job, outcome in zip(chunk, workers.run_batch(items, concurrency=BATCH), strict=True):
                text = extract_text(outcome["result"]) if outcome.get("ok") else ""
                if len(text.split()) < config.MIN_WORDS:
                    print("  no usable text:", job["id"], (outcome.get("error") or "short or empty")[:120])
                    continue
                handle.write(json.dumps({**job, "text": text}, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{min(start + BATCH, len(todo))}/{len(todo)}", flush=True)


def main() -> None:
    config.load_env()
    with open(config.raw_dir() / "raid" / "raid_sample.jsonl") as handle:
        humans = [row for row in map(json.loads, handle) if row["model"] == "human"]
    path = config.DATA_DIR / "generated.jsonl"
    workers = WorkersAI()
    run_jobs(workers, plan(humans, config.SEED), path, "generator")

    with open(path) as handle:
        existing = [json.loads(line) for line in handle]
    humanized_sources = {row["rewritten_from"] for row in existing if row["kind"] == "humanized"}
    fresh = [row for row in existing if row["kind"] == "fresh" and row["id"] not in humanized_sources]
    rng = random.Random(config.SEED + 1)
    rewrites = []
    for generator in GENERATORS:
        own = sorted((row for row in fresh if row["generator"] == generator), key=lambda row: row["id"])
        already = sum(1 for row in existing if row["kind"] == "humanized" and row["generator"] == generator)
        for n, row in enumerate(rng.sample(own, min(max(HUMANIZED_PER_GENERATOR - already, 0), len(own)))):
            candidates = [model for model in REWRITERS if model != generator]
            rewriter = candidates[n % len(candidates)]
            rewrites.append({
                "id": row["id"].replace("fresh-", "humanized-", 1), "kind": "humanized",
                "generator": generator, "rewriter": rewriter, "domain": row["domain"],
                "source_id": row["source_id"], "title": row["title"], "rewritten_from": row["id"],
                "prompt": HUMANIZE.format(text=row["text"]),
            })
    run_jobs(workers, rewrites, path, "rewriter")


if __name__ == "__main__":
    main()
