#!/usr/bin/env bash
# Keep the baseline queue fed: every 60 s select finished searches and queue their remaining seeds
# (scripts/select_and_queue.py), and keep N workers alive (N read from logs/n_workers, default 2).
#   nohup bash scripts/auto_queue.sh > logs/auto_queue.log 2>&1 &
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
Q="$ROOT/logs/queue.txt"
source "$ROOT/.venv/bin/activate"
while true; do
  python "$ROOT/scripts/select_and_queue.py" --queue "$Q" > "$ROOT/logs/select_last.log" 2>&1
  N=$(cat "$ROOT/logs/n_workers" 2>/dev/null || echo 2)
  running=$(pgrep -fc "scripts/run_queue.sh" || true)
  if [[ -s "$Q" ]] && (( running < N )); then
    id=$(date +%s)
    THREADS="${THREADS:-3}" nohup bash "$ROOT/scripts/run_queue.sh" "$Q" > "$ROOT/logs/worker_$id.log" 2>&1 &
    echo "[$(date '+%F %T')] launched worker $id (running=$running N=$N)"
  fi
  # stop when every method has all seeds and nothing is queued or running
  if ! grep -q "incomplete" "$ROOT/logs/select_last.log" && [[ ! -s "$Q" ]] && (( running == 0 )); then
    python "$ROOT/scripts/select_and_queue.py" --queue "$Q" | grep -q "queued 0 runs" && { echo "all done"; break; }
  fi
  sleep 60
done
