#!/usr/bin/env bash
# Point repo at user-provided Beauty LOO data (no copy of full TSV into git).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Usage: PCDREC_DATA_DIR=/path/to/amazon-beauty bash scripts/00_link_user_data.sh
# (directory produced by scripts/00_build_beauty_from_raw.sh or provided separately)
SRC="${PCDREC_DATA_DIR:-${PCDREC_DATA:-}}"
if [[ -z "$SRC" ]]; then
  echo "Set PCDREC_DATA_DIR to the Beauty data directory (train/valid/test/item_meta.txt + manifest.json)" >&2
  exit 1
fi
DST_RAW="$ROOT/data/raw/amazon-beauty"

if [[ ! -d "$SRC" ]]; then
  echo "ERROR: data dir not found: $SRC" >&2
  exit 1
fi

mkdir -p "$(dirname "$DST_RAW")"
if [[ -L "$DST_RAW" || -e "$DST_RAW" ]]; then
  rm -rf "$DST_RAW"
fi
ln -s "$SRC" "$DST_RAW"
echo "Linked $DST_RAW -> $SRC"
ls -la "$DST_RAW"
