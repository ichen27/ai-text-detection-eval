# ai-text-detection-eval

Repository: https://github.com/ichen27/ai-text-detection-eval (private).
Canonical checkout: `/Volumes/SamsungSSD1/code/ai-text-detection-eval` on the Mac mini SSD.

## What this is

A small, reproducible experiment on detecting AI-written text. It compiles labelled text sets and
compares three kinds of detector on them: decision models asked a typed question with no training
(Cloudflare Clef), statistical detectors that need no training (Binoculars, Fast-DetectGPT), and a
classifier fine-tuned on examples. It is an evaluation harness, not a detector product.

## Layout

- `src/detecteval/` — one module per pipeline stage; shared pieces in `config`, `hub`, `questions`,
  `metrics`, `textprep`, `items`.
  - First study (texts of 150 words or fewer): `fetch_raid`, `fetch_essays`, `generate`, `build_dataset`,
    `score`, `report`.
  - Second study (long documents at several lengths): `fetch_raid_full`, `fetch_stackexchange`,
    `build_long_set`, `generate_long`, `score_long`.
  - Local detectors: `binoculars`; `generate_mirrors`, `train_data`, `train_classifier`, `classify`.
  - Comparison of everything on held-out halves: `compare`, `compare_report`.
- `data/` — both evaluation sets, the texts generated for them, and the training mirrors (committed).
- `results/` — scores per detector, summaries, reports and figures (committed deliverables).
- Raw third-party downloads and trained model weights live outside Git in `RAW_DATA_DIR` (default
  `raw/`, ignored).

## Conventions

- Scores always mean "higher = more likely AI"; labels are `human`, `ai`, `mixed`.
- Every text passes through `build_dataset.prepare`, so groups are compared at the same length and layout.
- Sampling is seeded by `config.SEED`. Changing the seed, caps or prompts changes the dataset: rebuild,
  rescore and regenerate the report together, and never mix scores from different dataset versions.
- Model access goes through `hub.McpClient`. Tokens are read from a file named by `MCP_TOKEN_FILE`;
  never print, log or commit a token, and keep `.env` untracked.
- Scoring and generation call paid APIs. Keep runs resumable and report the token cost.
- A request through an MCP gateway may be cut off after 60 seconds. `hub.WorkersAI.run_batch` gives each
  item its own time limit so one slow item cannot lose the batch; keep single generations under that.
- A "key" is one text at one length: `<id>@short`, or `<id>@<words>` for the long set. Every detector
  writes one row per key, so `compare` can join them.
- Long-set texts carry a `split`. Set thresholds and fit combinations on `calib`; report on `test`.
- Training data must never overlap evaluation data. `train_data` excludes every evaluation source
  document, uses separate Stack Exchange sites, and leaves four current generators and all student
  essays out of training. Keep those exclusions when changing either side.
- The local detectors need the optional `ml` dependency group (`uv sync --group ml`) and several
  gigabytes of free memory. Run one at a time. CI does not install that group, so modules that need
  PyTorch import it inside `main()` and tests cover only the NumPy parts.

## Setup and validation

```sh
uv sync
bash ci/run.sh        # ruff + pytest; the same entry point CI uses
```

Tests cover the metrics, text preparation, half assignment and the detector statistics. They do not call
any model and do not need PyTorch.

## Workflow

Work on a feature or fix branch, push it, open a PR, and merge only after the `test` check passes on the
current revision. Do not push implementation changes straight to `main`. Commit as the repository owner
only; no tool or agent attribution in commits, PRs or files.
