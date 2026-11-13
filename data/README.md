# Data

| File | What it is |
|---|---|
| `eval_set.jsonl` | First evaluation set: 836 texts of at most 150 words, already trimmed and normalised. |
| `generated.jsonl` | Raw model outputs written for the first set (prompt, generator, untrimmed text). |
| `long_set.jsonl` | Second evaluation set: documents of 300 words or more, kept up to 500 words, so each can be scored at 75, 150, 300 and 500 words. Every text is assigned to a `calib` or `test` half. |
| `generated_long.jsonl` | Raw model outputs written for the second set. So far fresh texts only (79 of 160); humanized rewrites and polished human texts are planned. |
| `train_mirrors.jsonl` | Texts written by four current models on the titles of human training documents (264 of 1,000 planned). Used only to train the classifier; never part of an evaluation set. |

## Fields shared by both evaluation sets

- `id` — stable identifier.
- `label` — `human`, `ai`, or `mixed` (written by a person, then polished by a model).
- `group` — where the text comes from (see the tables in the top-level README).
- `source`, `domain`, `generator` — dataset, genre, and the model that wrote it (`human` for people).
- `rewriter` — for humanized and polished texts, the model that rewrote it.
- `source_id` — the source document; human and AI texts with the same value share a title.
- `words`, `text` — word count and the text as stored.

## Extra fields in `long_set.jsonl`

- `split` — `calib` or `test`. Thresholds and combination weights are set on `calib`; results are reported
  on `test`. Texts that share a `source_id` share a half.
- `title` — the title the text was written for (empty for essays).
- `site`, `link`, `author`, `created` — for forum posts: the Stack Exchange site, the post's URL, its
  author's display name and its date. These are the attribution required by CC BY-SA 4.0.
- `rewritten_from` appears in `generated_long.jsonl`: the id of the text a rewrite started from.

A text counts for a length when it has at least 85% of that many words. All documents fill the 75-, 150-
and 300-word lengths; fewer reach 500 words.

## Not in the repository

Raw third-party downloads are kept outside Git in `RAW_DATA_DIR` (default `raw/`):

| Download | Size | Fetched by |
|---|---|---|
| RAID parquet files, four domains, with adversarial attacks | 2.4 GB | `python -m detecteval.fetch_raid` |
| RAID full non-adversarial training split, eight domains | 0.8 GB (plus 0.4 GB as parquet) | `python -m detecteval.fetch_raid_full` |
| Essay sets from Liang et al. | under 1 MB | `python -m detecteval.fetch_essays` |
| Stack Exchange posts for evaluation and training | about 10 MB | `python -m detecteval.fetch_stackexchange` |
| Trained classifier weights | about 0.6 GB per model | `python -m detecteval.train_classifier` |

The Stack Exchange fetch takes "recent" to mean "from the day after the Clef models were released until
today", so running it later returns a different, larger sample. See `THIRD_PARTY_NOTICES.md` for sources
and licences.
