#!/usr/bin/env bash
# Train Stage-0 SASRec CE. Default: smoke subset on CPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

EPOCHS="${EPOCHS:-2}"
BATCH="${BATCH:-128}"
TAG="${TAG:-sasrec}"

ARGS=(
  --model-config "$ROOT/configs/model/sasrec.yaml"
  --data-config "$ROOT/configs/data/beauty.yaml"
  --epochs "$EPOCHS"
  --batch-size "$BATCH"
  --tag "$TAG"
  --skip-process
)

if [[ "${FULL:-0}" == "1" ]]; then
  echo "Training SASRec FULL users: epochs=$EPOCHS batch=$BATCH"
else
  MAX_USERS="${MAX_USERS:-1000}"
  ARGS+=(--max-users "$MAX_USERS")
  echo "Training SASRec smoke: max_users=$MAX_USERS epochs=$EPOCHS batch=$BATCH"
fi

python -m pcdrec.train "${ARGS[@]}"
