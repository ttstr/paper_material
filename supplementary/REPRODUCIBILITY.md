# Reproducibility — PCDRec

This document is supplementary material for the anonymous RecSys submission.
In the paper, the code link is the placeholder `https://anonymous.4open.science/r/PCDRec-XXXX` (TODO: create the anonymous mirror).
All commands below come from the repository `README.md` and are run from the repository root.

## 1. Status at a glance

| Component | Status | Evidence |
|---|---|---|
| Data loading + sha256/row-count/LOO checks | **Reproduced** | `scripts/01_load_splits.sh`, `tests/test_split.py` |
| Leakage guards (LLM inputs never contain valid/test targets; hold-out prefix) | **Implemented + tested** | `tests/test_no_leakage.py` |
| Full-catalog HR/NDCG evaluator | **Implemented + tested** | `tests/test_metrics.py` |
| Online export with no LLM imports and SASRec-only parameter count | **Implemented + tested** | `tests/test_online_no_llm.py` |
| Test suite | **22 passed** (re-run 2026-10-06 UTC+8, after data-build + baseline framework) | `pytest -q` |
| SASRec-CE baseline, Beauty, 5 seeds, CPU | **Reproduced (only real result)** | `results/main_table_sasrec_full.{md,csv}`, `results/sasrec_full_s4[2-6]_metrics.{json,csv}`, `logs/sasrec_full_s4[2-6].log` |
| SASRec smoke run (1000 users) | Smoke test only, **not a paper result** | `results/sasrec_subset1000_metrics.*` |
| Offline LLM profiling / teacher ranking | **TODO** (stub `src/pcdrec/llm_offline/`) | — |
| PCS verification (G, T, R, F) + negatives N1–N3 | **TODO** (stub `src/pcdrec/verify/`) | — |
| Text encoder / claim pooling | **TODO** (stub `src/pcdrec/encode/`) | — |
| Distillation losses L_align (S-DPO, ListKL) and L_pref | **TODO** | — |
| Baselines GRU4Rec, BERT4Rec, SASRec-T, UniSRec, KAR, DLLM2Rec, RDRec, Persona4Rec†, Ocean-feat†, Naive-Distill, LLM rerank reference | **TODO** (stub `src/pcdrec/baselines/`) | — |
| Ablations A1–A10, sensitivity, group analysis, case study | **TODO** | — |
| Latency benchmark (P50/P95, throughput) | **TODO** (no measurement yet) | — |
| Offline LLM cost accounting (calls, tokens, GPU-h) | **TODO** | — |
| Determinism test (`test_determinism.py` from the plan) | **TODO** (not in the repo yet) | — |
| One-command data rebuild from the official Zenodo archive with FreeRec 0.9.7 (RecBoard `Amazon2014Beauty_550_LOU`) + sha256 check | **Reproduced byte-for-byte** (2026-10-06, Linux, fresh venv) | `scripts/00_build_beauty_from_raw.sh`, `scripts/data_build/build_beauty_from_recboard.py`, DATA_STATEMENT.md §1/§3 |

## 2. Environment (machine that produced the baseline)

| Item | Value |
|---|---|
| OS / arch | Linux, x86_64 |
| CPU | Intel Xeon, 8 cores; ~15 GB RAM; **no GPU** |
| Python | 3.13.5 |
| torch | 2.14.1+cpu |
| numpy / pandas / pyarrow / PyYAML / tqdm / pytest | 2.5.3 / 3.0.6 / 25.0.1 / 6.0.3 / 4.70.1 / 9.1.1 |
| Threads | seed 42: 4 torch threads (run alone); seeds 43–46: 3 threads each, two parallel queues |

`requirements.txt` gives lower bounds; **`requirements-lock.txt`** is the exact `pip freeze` of the environment used for every run (Python 3.13.5, recorded in `.python-version`). The data build uses a separate environment pinned in `requirements-freerec.txt` (freerec 0.9.7, pandas 2.3.3, numpy 2.5.0, polars 1.44.2, torchdata 0.7.0 `--no-deps`).

## 3. Seeds and determinism
- Seeds: 42, 43, 44, 45, 46. `train.py:set_seed` seeds `random`, `numpy` and `torch` (plus CUDA when available).
- Results are mean ± sample std (ddof = 1) over the 5 seeds, aggregated by `scripts/06_tables.py`.
- CPU runs used different thread counts (see above). Floating-point results may differ slightly across thread counts and hardware.
- Planned: LLM decoding at temperature 0 with cached artifacts; prompt hash, token counts and model version logged for every call (TODO).

## 4. Exact commands (from the repository README)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cpu   # when there is no GPU
pip install -r requirements.txt
export PYTHONPATH=$PWD/src

# 1) rebuild the data from Zenodo with FreeRec 0.9.7 (+ sha256 check), then validate/export processed splits
bash scripts/00_build_beauty_from_raw.sh          # output dir: data/raw/amazon-beauty or $PCDREC_DATA_DIR
bash scripts/01_load_splits.sh

# 2) unit tests
pytest -q

# 3) CPU smoke run, 1000 users (not a paper result)
MAX_USERS=1000 EPOCHS=10 BATCH=128 bash scripts/04_train_sasrec.sh

# 4) evaluate / export the online model (no LLM)
bash scripts/05_eval.sh results/checkpoints/sasrec_subset1000_best.pt
python -m pcdrec.export_online --checkpoint results/checkpoints/sasrec_subset1000_best.pt --out-dir results/online_export

# 5) full SASRec-CE baseline (paper Table 3, row "SASRec-CE")
TQDM_DISABLE=1 FULL=1 EPOCHS=200 PATIENCE=20 BATCH=256 THREADS=4 SEED=42 TAG=sasrec_full_s42 \
  bash scripts/04_train_sasrec.sh > logs/sasrec_full_s42.log 2>&1
THREADS=3 nohup bash scripts/run_sasrec_full_seeds.sh 43 45 > logs/queue_a.log 2>&1 &
THREADS=3 nohup bash scripts/run_sasrec_full_seeds.sh 44 46 > logs/queue_b.log 2>&1 &

# 6) aggregate the per-seed JSONs into the main table
python scripts/06_tables.py
```
Paths: all configs use repository-relative paths (`data/raw/amazon-beauty`, `data/processed/beauty`, `results/`). `PCDREC_DATA_DIR` overrides the raw data directory and `PCDREC_PROCESSED_DIR` the processed cache; no machine-specific absolute path is needed.

## 5. Expected output (baseline)
`results/main_table_sasrec_full.md` (auto-generated; do not hand-edit). Test, mean ± std over 5 seeds:

| HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|---|---|---|---|---|---|
| 0.0504 ± 0.0010 | 0.0763 ± 0.0027 | 0.1089 ± 0.0053 | 0.0327 ± 0.0010 | 0.0410 ± 0.0004 | 0.0492 ± 0.0006 |

Protocol: 22,363 users; full-catalog ranking over 12,101 items; history items **not** filtered; valid uses train history, test uses train+valid history; early stopping on valid NDCG@10 with patience 20; params 877,824. Run cost: 1.61 h total wall time over the 5 seeds (CPU).

## 6. Model / training configuration
`configs/model/sasrec.yaml`: 2 layers, 2 heads, hidden 64, FFN 256, max_len 50, dropout 0.2 (hidden and attention), GELU, init range 0.02, full-softmax CE, Adam lr 1e-3, weight decay 0, batch 256, at most 200 epochs, patience 20.
Planned PCDRec defaults (configuration, not results): M=20, P=3, h=3, m=5, β∈{0.5,1,2}, λ1,λ2∈{0.01,0.1,0.5,1}, τ∈{0.05,0.1}. `configs/method/pcdrec.yaml` is currently a placeholder (`stage: 0`).

## 7. What is not shipped
Raw or processed data, checkpoints (`*.pt`), online exports, full LLM output caches, and model weights (LLM/NLI/encoder) are git-ignored. See DATA_STATEMENT.md.
