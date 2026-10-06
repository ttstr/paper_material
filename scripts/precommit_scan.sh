#!/usr/bin/env bash
# Scan staged files for secrets, personal paths and raw data before committing/pushing.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
fail=0
files=$(git diff --cached --name-only --diff-filter=ACMR)
[[ -z "$files" ]] && { echo "nothing staged"; exit 0; }
SECRET='(sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|(api[_-]?key|secret|token)["'"'"']?\s*[:=]\s*["'"'"'][A-Za-z0-9_\-]{16,}["'"'"'])'
PERSONAL='(C:\\\\Users|C:/Users|wyphm|miniconda3|D:\\\\|/home/[a-z]+/|/Users/[A-Za-z]+/)'
RAWNAME='(\.inter$|\.item$|\.zip$|\.tsv$|\.parquet$|\.pkl$|\.pt$|\.pth$|\.ckpt$|\.safetensors$|\.bin$|\.npy$|(^|/)(train|valid|test|item_meta|item)\.txt$|\.env$)'
for f in $files; do
  [[ -f "$f" ]] || continue
  if echo "$f" | grep -Eq "$RAWNAME"; then echo "RAW/WEIGHT FILE: $f"; fail=1; fi
  sz=$(stat -c %s "$f"); if (( sz > 5000000 )); then echo "LARGE FILE (>5MB): $f"; fail=1; fi
  if grep -IEn "$SECRET" "$f" >/dev/null 2>&1; then echo "POSSIBLE SECRET: $f"; grep -IEn "$SECRET" "$f" | head -3; fail=1; fi
  if [[ "$f" != scripts/precommit_scan.sh ]] && grep -IEn "$PERSONAL" "$f" >/dev/null 2>&1; then echo "PERSONAL PATH: $f"; grep -IEn "$PERSONAL" "$f" | head -3; fail=1; fi
  # Amazon raw records: reviewer-id like tokens followed by ASIN
  if grep -IEq "^A[0-9A-Z]{12,14}\s+[0-9A-Z]{10}\s" "$f"; then echo "RAW AMAZON RECORDS: $f"; fail=1; fi
done
(( fail )) && { echo "scan FAILED"; exit 1; } || echo "scan OK ($(echo "$files" | wc -l) files)"
