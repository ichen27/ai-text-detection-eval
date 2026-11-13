"""Generate the machine-written and machine-polished texts of the long set with current Workers AI models.

    python -m detecteval.build_long_set     # choose the human sources first
    python -m detecteval.generate_long
    python -m detecteval.build_long_set     # then merge the generated texts in

Three kinds of text are written to data/generated_long.jsonl (finished ids are skipped on a rerun):
  fresh      about 550 words on the title of a sampled human document, in the same genre
  humanized  a second model rewrites a fresh text to sound like an ordinary person
  polished   a model tidies a human forum post or student essay (reported as "mixed", not AI)
"""
from __future__ import annotations

import json
import random

from . import config
from .generate import GENERATORS, HUMANIZE, REWRITERS, run_jobs, short_name
from .hub import WorkersAI

PER_GENERATOR = {"news": 3, "books": 3, "poetry": 2, "recipes": 2, "reviews": 2, "reddit": 1, "wiki": 1,
                 "forum": 4, "essay": 2}
HUMANIZED_PER_GENERATOR = 6
POLISHED = {"forum_2019": 40, "essay_native": 20}
MAX_TOKENS = 4500
BATCH = 6   # a Worker holds at most six connections open at once
# The MCP gateway ends any request after 60 seconds. At this length four models only finish in time
# with their hidden reasoning switched off, and the slowest model is asked for fewer words.
NO_REASONING = {
    "@cf/qwen/qwen3.8-27b": {"chat_template_kwargs": {"enable_thinking": False}},
    "@cf/google/gemma-4-26b-a4b-it": {"chat_template_kwargs": {"enable_thinking": False}},
    "@cf/moonshotai/kimi-k2.6": {"chat_template_kwargs": {"thinking": False}},
    "@cf/deepseek-ai/deepseek-v4.1-flash-1017": {"chat_template_kwargs": {"thinking": False}},
}
WORD_TARGET = {"@cf/meta/llama-4-scout-17b-16e-instruct": 380}

PROMPTS = {
    "news": 'Write the body of a BBC news article titled "{title}". Do not repeat the title.',
    "books": 'Write the body of a plot summary for a novel titled "{title}". Do not give it a title.',
    "poetry": 'Write the body of a poem titled "{title}". Do not repeat the title.',
    "recipes": 'Write a recipe for "{title}".',
    "reviews": 'Write the body of an IMDb review for the movie "{title}". Do not give it a title.',
    "reddit": 'Write just the body of a Reddit post titled "{title}". Do not repeat the title.',
    "wiki": 'Write the body of a Wikipedia article titled "{title}".',
    "forum": 'Write just the body of a question posted on the "{site}" Stack Exchange site, titled "{title}". '
             "Do not repeat the title.",
    # The task the native-speaker student essays were written for (ASAP essay set 1), paraphrased.
    "essay": "Some people believe computers benefit society: they teach hand-eye coordination, let people learn "
             "about faraway places and talk with others online. Others worry that people spend too much time on "
             "computers and too little exercising, enjoying nature and being with family and friends. Write a "
             "letter to your local newspaper giving your opinion on the effects computers have on people, and "
             "persuade the readers to agree with you.",
}
LENGTH_NOTE = " Write about {words} words. Output only the text itself, without any commentary."
POLISH = ("Improve the writing of the following text. Fix grammar and spelling and make the wording clearer and "
          "more fluent, but keep the meaning, the structure, the voice and the length. Output only the improved "
          "text.\n\n{text}")


def fresh_jobs(humans: list[dict], seed: int) -> list[dict]:
    """Give each generator its own human source documents in every genre."""
    rng = random.Random(seed + 10)
    jobs = []
    for genre, per_generator in PER_GENERATOR.items():
        if genre == "essay":
            sources = [{"source_id": f"essay-prompt-{n}", "title": "", "site": ""} for n in range(per_generator)]
            sources = [sources] * len(GENERATORS)
        else:
            group = "forum_recent" if genre == "forum" else "raid_human"
            pool = sorted((h for h in humans if h["group"] == group and h["domain"] == genre and h["title"].strip()),
                          key=lambda h: h["id"])
            rng.shuffle(pool)
            sources = [pool[g * per_generator:(g + 1) * per_generator] for g in range(len(GENERATORS))]
        for generator, own in zip(GENERATORS, sources, strict=True):
            for source in own:
                prompt = PROMPTS[genre].format(title=source["title"].strip(), site=source.get("site", ""))
                jobs.append({"id": f"fresh-{short_name(generator)}-{source['source_id'][-12:]}", "kind": "fresh",
                             "generator": generator, "domain": genre, "source_id": source["source_id"],
                             "title": source["title"].strip(),
                             "prompt": prompt + LENGTH_NOTE.format(words=WORD_TARGET.get(generator, 550))})
    return jobs


def humanize_jobs(fresh: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed + 11)
    jobs = []
    for generator in GENERATORS:
        own = sorted((row for row in fresh if row["generator"] == generator), key=lambda row: row["id"])
        for n, row in enumerate(rng.sample(own, min(HUMANIZED_PER_GENERATOR, len(own)))):
            candidates = [model for model in REWRITERS if model != generator]
            jobs.append({"id": row["id"].replace("fresh-", "humanized-", 1), "kind": "humanized",
                         "generator": generator, "rewriter": candidates[n % len(candidates)],
                         "domain": row["domain"], "source_id": row["source_id"], "title": row["title"],
                         "rewritten_from": row["id"], "prompt": HUMANIZE.format(text=row["text"])})
    return jobs


def polish_jobs(humans: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed + 12)
    jobs = []
    for group, wanted in POLISHED.items():
        pool = sorted((h for h in humans if h["group"] == group), key=lambda h: h["id"])
        for n, human in enumerate(rng.sample(pool, wanted)):
            jobs.append({"id": "polished-" + human["id"], "kind": "polished", "generator": "human",
                         "rewriter": REWRITERS[n % len(REWRITERS)], "domain": human["domain"],
                         "source_id": human["source_id"], "title": human["title"], "rewritten_from": human["id"],
                         "prompt": POLISH.format(text=human["text"])})
    return jobs


def main() -> None:
    config.load_env()
    with open(config.DATA_DIR / "long_set.jsonl") as handle:
        humans = [row for row in map(json.loads, handle) if row["label"] == "human"]
    path = config.DATA_DIR / "generated_long.jsonl"
    workers = WorkersAI()
    run_jobs(workers, fresh_jobs(humans, config.SEED), path, "generator", MAX_TOKENS, BATCH, NO_REASONING)
    with open(path) as handle:
        fresh = [row for row in map(json.loads, handle) if row["kind"] == "fresh"]
    run_jobs(workers, humanize_jobs(fresh, config.SEED), path, "rewriter", MAX_TOKENS, 2)
    run_jobs(workers, polish_jobs(humans, config.SEED), path, "rewriter", MAX_TOKENS, 2)


if __name__ == "__main__":
    main()
