#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

CKPT="${1:-}"
if [[ -z "$CKPT" ]]; then
  CKPT="$(ls -1t "$ROOT"/results/checkpoints/*_best.pt 2>/dev/null | head -1 || true)"
fi
if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
  echo "Usage: $0 <checkpoint.pt>" >&2
  exit 1
fi
OUT="${OUT:-$ROOT/results/eval_$(basename "$CKPT" .pt).json}"
python -m pcdrec.evaluate --checkpoint "$CKPT" --out "$OUT"
