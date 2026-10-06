#!/usr/bin/env bash
# Simple file-based job queue for CPU training. Each line of the queue file is
#   <tag> <model> <seed> [extra train.py args...]
# Lines are popped atomically (flock); append lines at any time. Launch N workers:
#   THREADS=3 nohup bash scripts/run_queue.sh logs/queue.txt > logs/worker_a.log 2>&1 &
#   THREADS=3 nohup bash scripts/run_queue.sh logs/queue.txt > logs/worker_b.log 2>&1 &
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
Q="${1:?queue file}"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TQDM_DISABLE=1
mkdir -p "$ROOT/logs"
while true; do
  line="$(flock "$Q.lock" bash -c 'l=$(head -n1 "'"$Q"'"); [ -n "$l" ] && sed -i 1d "'"$Q"'"; echo "$l"')"
  [[ -z "$line" ]] && break
  read -r tag model seed extra <<<"$line"
  if [[ -f "$ROOT/results/${tag}_metrics.json" ]]; then echo "skip $tag (exists)"; continue; fi
  echo "[$(date '+%F %T')] start $tag"
  # shellcheck disable=SC2086
  python -m pcdrec.train --model-config "$ROOT/configs/model/$model.yaml" --seed "$seed" \
    --threads "${THREADS:-3}" --tag "$tag" --skip-process $extra > "$ROOT/logs/$tag.log" 2>&1
  echo "[$(date '+%F %T')] done $tag rc=$?"
done
echo "[$(date '+%F %T')] queue empty"
