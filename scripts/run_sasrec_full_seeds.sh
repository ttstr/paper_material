#!/usr/bin/env bash
# Run full SASRec CE for a list of seeds sequentially (one queue). Launch 2 queues in parallel on 8-core CPU:
#   THREADS=3 nohup bash scripts/run_sasrec_full_seeds.sh 43 45 > logs/queue_a.log 2>&1 &
#   THREADS=3 nohup bash scripts/run_sasrec_full_seeds.sh 44 46 > logs/queue_b.log 2>&1 &
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/logs"
for s in "$@"; do
  echo "[$(date '+%F %T')] start seed=$s"
  TQDM_DISABLE=1 FULL=1 EPOCHS="${EPOCHS:-200}" BATCH="${BATCH:-256}" PATIENCE="${PATIENCE:-20}" \
    THREADS="${THREADS:-4}" SEED="$s" TAG="sasrec_full_s$s" \
    bash "$ROOT/scripts/04_train_sasrec.sh" > "$ROOT/logs/sasrec_full_s$s.log" 2>&1
  echo "[$(date '+%F %T')] done seed=$s"
done
