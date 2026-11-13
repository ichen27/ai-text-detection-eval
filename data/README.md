# Data

| File | What it is |
|---|---|
| `eval_set.jsonl` | The compiled evaluation set: one JSON object per text, already trimmed and normalised. This is what the models score. |
| `generated.jsonl` | Raw outputs generated for this study (prompt, generator, untrimmed text), before trimming. |

## `eval_set.jsonl` fields

- `id` — stable identifier.
- `label` — `human`, `ai`, or `mixed` (human-written, then polished by GPT-4).
- `group` — where the text comes from (see the table in the top-level README).
- `source`, `domain`, `generator` — dataset, genre, and the model that wrote it (`human` for people).
- `rewriter` — for humanized texts, the model that rewrote it.
- `source_id` — RAID source document; human and AI texts with the same value share a title.
- `words`, `text` — word count and the text as scored.

## Not in the repository

Raw third-party downloads (about 2.4 GB of RAID parquet files and the original essay JSON files) are
kept outside Git. `python -m detecteval.fetch_raid` and `python -m detecteval.fetch_essays` fetch them
again into `RAW_DATA_DIR` (default `raw/`). See `THIRD_PARTY_NOTICES.md` for sources and licences.
