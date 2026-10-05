#!/usr/bin/env bash
# Point repo at user-provided Beauty LOO data (no copy of full TSV into git).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${PCDREC_DATA:-/workspace/pcdrec-data/amazon-beauty}"
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
