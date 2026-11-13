#!/usr/bin/env bash
# Single local and CI entry point: lint and unit tests. Calls no model and needs no credentials.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || python -m pip install --quiet uv
uv sync --locked
uv run ruff check src tests
uv run pytest -q
