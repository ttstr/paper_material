#!/usr/bin/env bash
# One-command rebuild of the Beauty data (RecBoard Amazon2014Beauty_550_LOU) from the official
# Zenodo archive with FreeRec 0.9.7, then sha256 verification against the reference files.
#
#   bash scripts/00_build_beauty_from_raw.sh                  # -> data/raw/amazon-beauty
#   PCDREC_DATA_DIR=/data/beauty bash scripts/00_build_beauty_from_raw.sh
#   bash scripts/00_build_beauty_from_raw.sh --archive /path/Amazon2014Beauty.zip   # offline
#
# Env: PCDREC_DATA_DIR (output dir), FREEREC_VENV (default .freerec-venv), PCDREC_BUILD_DIR
# (scratch, default data/recboard_build). Extra args go to build_beauty_from_recboard.py.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${PCDREC_DATA_DIR:-$ROOT/data/raw/amazon-beauty}"
WORK="${PCDREC_BUILD_DIR:-$ROOT/data/recboard_build}"
VENV="${FREEREC_VENV:-$ROOT/.freerec-venv}"
PY="${PYTHON:-python3}"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[setup] creating FreeRec environment at $VENV"
  if command -v uv >/dev/null 2>&1; then
    uv venv -q -p "$PY" "$VENV"
    uv pip install -q --python "$VENV/bin/python" -r "$ROOT/requirements-freerec.txt" torch \
      --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
    uv pip install -q --python "$VENV/bin/python" torchdata==0.7.0 --no-deps
  else
    "$PY" -m venv "$VENV"
    "$VENV/bin/pip" install -q -U pip
    "$VENV/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cpu
    "$VENV/bin/pip" install -q -r "$ROOT/requirements-freerec.txt"
    "$VENV/bin/pip" install -q torchdata==0.7.0 --no-deps
  fi
fi

"$VENV/bin/python" "$ROOT/scripts/data_build/build_beauty_from_recboard.py" \
  --out-dir "$OUT" --work-dir "$WORK" --freerec-python "$VENV/bin/python" "$@"
echo "Data ready in $OUT. Next: bash scripts/01_load_splits.sh  (honours PCDREC_DATA_DIR)"
