#!/usr/bin/env bash
# CPU pilot, after LLM generation: cached replay -> PCS verify -> students (PCDRec, A1 naive) -> export -> check -> report.
#   nohup bash scripts/run_pilot_post.sh [wait_pid] > logs/pilot_post.log 2>&1 &
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
source .venv/bin/activate; export PYTHONPATH="$ROOT/src" TQDM_DISABLE=1
LLM=results/pilot/llm_qwen1.5b; CFG=configs/llm/qwen2.5-1.5b-cpu.yaml; T=${THREADS:-3}
if [[ -n "${1:-}" ]]; then while kill -0 "$1" 2>/dev/null; do sleep 30; done; fi
echo "[$(date '+%F %T')] llm generation finished"
python -m pcdrec.llm_offline.run_offline --config $CFG --n-users 60 --out $LLM --cached-only > logs/pilot_replay.log 2>&1
echo "[$(date '+%F %T')] replay done"; wc -l $LLM/*.jsonl
OMP_NUM_THREADS=$T python -m pcdrec.verify.run_verify --llm-dir $LLM > logs/pilot_verify.log 2>&1
echo "[$(date '+%F %T')] verify done"
python -m pcdrec.distill.train_student --llm-dir $LLM --tag pcdrec_pilot_s42 --seed 42 --threads $T --results-dir results/pilot > logs/pilot_student.log 2>&1
echo "[$(date '+%F %T')] student done"
python -m pcdrec.export_online --student --checkpoint results/checkpoints/pcdrec_pilot_s42_best.pt --out-dir results/pilot/online_export > logs/pilot_export.log 2>&1
python scripts/check_online_export.py --export-dir results/pilot/online_export --student-metrics results/pilot/pcdrec_pilot_s42_metrics.json --out results/pilot >> logs/pilot_export.log 2>&1
echo "[$(date '+%F %T')] export + check done"
python -m pcdrec.distill.train_student --llm-dir $LLM --tag pilot_a1_naive_s42 --seed 42 --threads $T --results-dir results/pilot \
  --ablation configs/ablation/A1_naive_distill.yaml > logs/pilot_student_a1.log 2>&1
echo "[$(date '+%F %T')] A1 student done"
python scripts/pilot_report.py --llm-dir $LLM --out results/pilot > logs/pilot_report.log 2>&1
echo "[$(date '+%F %T')] report done"
