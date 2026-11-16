# Proofmark

Repository: https://github.com/ichen27/proofmark (public; it was named ai-text-detection-eval until
2026-11-16, and the old address redirects).
Canonical checkout: `/Volumes/SamsungSSD1/code/ai-text-detection-eval` on the Mac mini SSD. The directory
and the Python package (`detecteval`) keep their original names.

## Shipping decisions

Recorded 2026-11-16.

- **Launcher:** yes. Deployed to the app launcher as `proofmark`.
- **Repository visibility:** public, MIT licence for the code.
- **Name:** brand name, Proofmark, on GitHub and the launcher.
- **Announcement:** X only. Post link: not posted yet.

The repository is public, so nothing private may be committed: no texts, titles, ids or per-document
scores from private writing, and no results for documents whose authorship is uncertain (use
`private_eval report --omit-group`).

## What this is

A local AI-text detector (web app and MCP server) and the reproducible evaluation behind it. The
evaluation compiles labelled text sets and compares three kinds of detector on them: decision models
asked a typed question with no training (Cloudflare Clef), statistical detectors that need no training
(Binoculars, Fast-DetectGPT), and a classifier fine-tuned on examples. The app uses the classifier.

## Layout

- `src/detecteval/` — one module per pipeline stage; shared pieces in `config`, `hub`, `questions`,
  `metrics`, `textprep`, `items`.
  - First study (texts of 150 words or fewer): `fetch_raid`, `fetch_essays`, `generate`, `build_dataset`,
    `score`, `report`.
  - Second study (long documents at several lengths): `fetch_raid_full`, `fetch_stackexchange`,
    `build_long_set`, `generate_long`, `score_long`.
  - Local detectors: `binoculars`; `generate_mirrors`, `train_data`, `train_classifier`, `classify`.
  - Comparison of everything on held-out halves: `compare`, `compare_report`.
  - Private writing: `private_eval` scores documents listed in `RAW_DATA_DIR/private/manifest.jsonl`.
  - Detector service: `calibration` (reference scores), `detector` (passages and bands, no PyTorch),
    `worker` (model process), `serve` (web page, JSON API and MCP endpoint), `static/index.html`.
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
- `generate_long` plans humanized rewrites from the fresh texts that exist when it runs. To retry failed
  fresh texts later, rerun only the fresh jobs, or the rewrite sample changes.
- Private documents, their titles, ids and per-document scores never enter the repository. Only the
  aggregate report from `private_eval report` is committed.
- Long-set texts carry a `split`. Set thresholds and fit combinations on `calib`; report on `test`.
- Training data must never overlap evaluation data. `train_data` excludes every evaluation source
  document, uses separate Stack Exchange sites, and leaves four current generators and all student
  essays out of training. Keep those exclusions when changing either side.
- The local detectors need the optional `ml` dependency group (`uv sync --group ml`) and several
  gigabytes of free memory. Run one at a time. CI does not install that group, so modules that need
  PyTorch import it inside `main()` and tests cover only the NumPy parts.

## The deployed detector

The service is deployed on the Mac mini through the app launcher as `proofmark`
(https://app-proofmark.chenagent.com, owner sign-in) from [launcher.json](launcher.json), and is
registered in the MCP hub as the eager HTTP backend `proofmark` (tools prefixed `proofmark__`).

- It runs from this canonical checkout with its own environment at
  `/Volumes/SamsungSSD1/tools/ai-text-detection/venv`, so `ci/run.sh` and `uv sync` here do not disturb
  it. Code changes take effect on restart: `node src/cli.mjs restart proofmark` in the launcher checkout.
  Keep the checkout on `main` when not actively working, or develop service changes in a worktree.
- After changing dependencies, redeploy with the launcher's `deploy` command so the build step runs.
- The service must stay light: models load in the worker on demand and the worker is stopped when idle.
  This machine has restarted from memory exhaustion before; do not make the models resident.
- Labels describe the score ("AI-like"), never the author ("written by AI"), and every result comes with
  how often human writing scores that high. The percentage is an equal-prior likelihood from
  `calibration.probability_curve`; keep the private human reference in that curve when it exists, and
  recheck the held-out calibration figures in the README if the classifier or reference scores change.
- `results/calibration.json` is derived from committed scores; rebuild it with `calibration` whenever the
  long-set scores or the classifier change. The private reference lives in `RAW_DATA_DIR/private`.

## Setup and validation

```sh
uv sync
bash ci/run.sh        # ruff + pytest; the same entry point CI uses
```

`bash ci/run.sh` syncs the default dependency groups, which removes PyTorch from the environment; run
`uv sync --group ml` again before the local detectors.

Tests cover the metrics, text preparation, half assignment and the detector statistics. They do not call
any model and do not need PyTorch.

## Workflow

Work on a feature or fix branch, push it, open a PR, and merge only after the `test` check passes on the
current revision. Do not push implementation changes straight to `main`. Commit as the repository owner
only; no tool or agent attribution in commits, PRs or files.
