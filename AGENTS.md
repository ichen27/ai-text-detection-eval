# ai-text-detection-eval

Repository: https://github.com/ichen27/ai-text-detection-eval (private).
Canonical checkout: `/Volumes/SamsungSSD1/code/ai-text-detection-eval` on the Mac mini SSD.

## What this is

A small, reproducible experiment: can general-purpose decision models tell human writing from AI writing
with no training? It compiles a labelled text set, asks each model typed questions, and reports
separation, false positives and calibration. It is an evaluation harness, not a detector product.

## Layout

- `src/detecteval/` — one module per pipeline stage: `fetch_raid`, `fetch_essays`, `generate`,
  `build_dataset`, `score`, `report`; shared pieces in `config`, `hub`, `questions`, `metrics`, `textprep`.
- `data/` — the compiled evaluation set and the texts generated for this study (committed).
- `results/` — per-model scores, summary, report and figures (committed deliverables).
- Raw third-party downloads live outside Git in `RAW_DATA_DIR` (default `raw/`, ignored).

## Conventions

- Scores always mean "higher = more likely AI"; labels are `human`, `ai`, `mixed`.
- Every text passes through `build_dataset.prepare`, so groups are compared at the same length and layout.
- Sampling is seeded by `config.SEED`. Changing the seed, caps or prompts changes the dataset: rebuild,
  rescore and regenerate the report together, and never mix scores from different dataset versions.
- Model access goes through `hub.McpClient`. Tokens are read from a file named by `MCP_TOKEN_FILE`;
  never print, log or commit a token, and keep `.env` untracked.
- Scoring and generation call paid APIs. Keep runs resumable and report the token cost.

## Setup and validation

```sh
uv sync
bash ci/run.sh        # ruff + pytest; the same entry point CI uses
```

Tests cover the metrics and text preparation only. They do not call any model.

## Workflow

Work on a feature or fix branch, push it, open a PR, and merge only after the `test` check passes on the
current revision. Do not push implementation changes straight to `main`. Commit as the repository owner
only; no tool or agent attribution in commits, PRs or files.
