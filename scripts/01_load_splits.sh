#!/usr/bin/env bash
# Validate manifest + build processed sequences.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pcdrec.data.load_splits --config "$ROOT/configs/data/beauty.yaml"
