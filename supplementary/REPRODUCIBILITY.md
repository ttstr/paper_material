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
| Test suite | **47 passed** (re-run 2026-10-06 21:10 UTC+8) | `pytest -q` |
| SASRec-CE baseline, Beauty, 5 seeds, CPU | **Reproduced** | `results/main_table_sasrec_full.{md,csv}`, `results/sasrec_full_s4[2-6]_metrics.{json,csv}`, `logs/sasrec_full_s4[2-6].log` |
| Non-LLM baselines GRU4Rec (CE), BERT4Rec (MLM), SASRec-T (frozen MiniLM text features), MF-BPR; same evaluator, same capacity, 2-config budget each, 5 seeds | **Run** (see table for the seeds actually completed) | `results/main_table_beauty.{md,csv}`, `results/search_summary.csv`, `configs/search/baselines.yaml`, `logs/<tag>.log` |
| Paired significance tests vs SASRec (per-user test NDCG@10; Wilcoxon + paired t, Holm) | **Run** | `results/significance_beauty.json`, `results/main_table_beauty.md` |
| SASRec smoke run (1000 users) | Smoke test only, **not a paper result** | `results/sasrec_subset1000_metrics.*` |
| Offline LLM profiling / teacher ranking (transformers / vLLM / OpenAI-compatible backends; JSON schema; resumable cache; per-call prompt hash, tokens, seconds) | **Implemented + tested**; CPU pilot run with Qwen2.5-1.5B-Instruct | `src/pcdrec/llm_offline/`, `tests/test_llm_offline.py`, `results/pilot/` |
| PCS verification (G via NLI, T temporal AUC, R Kendall-τ, F faithfulness) + weights + negatives N1–N3 | **Implemented + tested**; run on the pilot users | `src/pcdrec/verify/`, `tests/test_verify.py` |
| Distillation losses L_align (S-DPO with frozen reference, ListKL, BPR) and L_pref (InfoNCE, AttnPool) | **Implemented + tested** | `src/pcdrec/losses/`, `tests/test_losses.py` |
| Student training (init from Stage-0), ablation switches A1–A6, online export (backbone only) | **Implemented + tested**; pilot run | `src/pcdrec/distill/`, `configs/ablation/`, `scripts/check_online_export.py` |
| CPU pilot (small model, small user subset): end-to-end, throughput, full-cost estimate | **Pilot only, not a paper result** | `results/pilot/PILOT_REPORT.md`, `results/pilot/pilot_summary.json` |
| 7B-teacher full offline run, PCDRec main result, LLM-based baselines (KAR, DLLM2Rec, RDRec, UniSRec, Persona4Rec†, Ocean-feat†, LLM rerank reference) | **TODO** (needs GPU or API budget) | — |
| Ablations A7–A10, sensitivity, group analysis, case study | **TODO** | — |
| Latency benchmark | Single-request CPU latency of the exported model measured in the pilot export check; full P50/P95 benchmark **TODO** | `results/pilot/online_export_check.json` |
| Determinism test (`test_determinism.py` from the plan) | **TODO** | — |
| One-command data rebuild from the official Zenodo archive with FreeRec 0.9.7 (RecBoard `Amazon2014Beauty_550_LOU`) + sha256 check | **Reproduced byte-for-byte** (2026-10-06, Linux, fresh venv) | `scripts/00_build_beauty_from_raw.sh`, `scripts/data_build/build_beauty_from_recboard.py`, DATA_STATEMENT.md §1/§3 |

## 2. Environment (machine that produced the baseline)

| Item | Value |
|---|---|
| OS / arch | Linux, x86_64 |
| CPU | Intel Xeon, 8 cores; ~15 GB RAM; **no GPU** |
| Python | 3.13.5 |
| torch | 2.14.1+cpu |
| numpy / pandas / pyarrow / PyYAML / tqdm / pytest | 2.5.3 / 3.0.6 / 25.0.1 / 6.0.3 / 4.70.1 / 9.1.1 |
| Threads | SASRec seed 42: 4 torch threads (run alone); every other baseline run: 3 threads, 2–3 parallel workers (recorded per run as `torch_threads`) |

`requirements.txt` gives lower bounds; **`requirements-lock.txt`** is the exact `pip freeze` of the environment used for every run (Python 3.13.5, recorded in `.python-version`). The data build uses a separate environment pinned in `requirements-freerec.txt` (freerec 0.9.7, pandas 2.3.3, numpy 2.5.0, polars 1.44.2, torchdata 0.7.0 `--no-deps`).

## 3. Seeds and determinism
- Seeds: 42, 43, 44, 45, 46. `train.py:set_seed` seeds `random`, `numpy` and `torch` (plus CUDA when available).
- Results are mean ± sample std (ddof = 1) over the 5 seeds, aggregated by `scripts/06_tables.py`.
- CPU runs used different thread counts (see above). Floating-point results may differ slightly across thread counts and hardware.
- LLM decoding is greedy (temperature 0); every call is cached in `calls.jsonl` with prompt hash, model name, decoding parameters, token counts and seconds; `--cached-only` rebuilds all LLM artefacts from the cache without loading a model.

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

# 5) full baselines: seed-42 search (2 configs per method), then seeds 42-46 of the selected config
python -m pcdrec.encode.text_encoder                       # MiniLM item-text cache for SASRec-T
bash scripts/auto_queue.sh                                 # feeds logs/queue.txt, keeps N workers (logs/n_workers)

# 6) main table + paired significance tests
python scripts/06_tables.py

# 7) CPU pilot of the LLM pipeline (small model, small subset; not a paper result)
python -m pcdrec.llm_offline.run_offline --config configs/llm/qwen2.5-1.5b-cpu.yaml --n-users 60 --out results/pilot/llm_qwen1.5b
python -m pcdrec.verify.run_verify --llm-dir results/pilot/llm_qwen1.5b
python -m pcdrec.distill.train_student --llm-dir results/pilot/llm_qwen1.5b --tag pcdrec_pilot_s42 --results-dir results/pilot
python -m pcdrec.export_online --student --checkpoint results/checkpoints/pcdrec_pilot_s42_best.pt --out-dir results/pilot/online_export
python scripts/check_online_export.py --export-dir results/pilot/online_export --student-metrics results/pilot/pcdrec_pilot_s42_metrics.json
python scripts/pilot_report.py
```
Paths: all configs use repository-relative paths (`data/raw/amazon-beauty`, `data/processed/beauty`, `results/`). `PCDREC_DATA_DIR` overrides the raw data directory and `PCDREC_PROCESSED_DIR` the processed cache; no machine-specific absolute path is needed.

## 5. Expected output
- `results/main_table_beauty.md` (auto-generated; do not hand-edit): test mean ± std over seeds for every baseline, the selected configuration, train time per seed, the tuning record, and paired tests against SASRec.
- `results/main_table_sasrec_full.md`: the SASRec-only table (`python scripts/06_tables.py --glob 'results/sasrec_full_s*_metrics.json' --out-stem results/main_table_sasrec_full`).
- `results/pilot/PILOT_REPORT.md`: the CPU pilot (not a paper result).

Protocol: 22,363 users; full-catalog ranking over 12,101 items; history items **not** filtered; valid uses train history, test uses train+valid history; early stopping on valid NDCG@10 with patience 20.

## 6. Model / training configuration
`configs/model/sasrec.yaml`: 2 layers, 2 heads, hidden 64, FFN 256, max_len 50, dropout 0.2 (hidden and attention), GELU, init range 0.02, full-softmax CE, Adam lr 1e-3, weight decay 0, batch 256, at most 200 epochs, patience 20.
Other baselines: `configs/model/{gru4rec,bert4rec,sasrec_t,mf_bpr}.yaml`, with the search grid in `configs/search/baselines.yaml`.
PCDRec defaults (configuration, not results): `configs/method/pcdrec.yaml` (M=20, P=3, h=3, K≤8, m=5, S-DPO β=1, λ1=λ2=0.1, τ=0.1, student initialised from Stage-0 SASRec seed 42); teacher configs in `configs/llm/` (paper default `qwen2.5-7b.yaml`; CPU pilot `qwen2.5-1.5b-cpu.yaml`).

## 7. What is not shipped
Raw or processed data (including the cached item-text embeddings), checkpoints (`*.pt`), online exports, raw LLM call caches (`calls.jsonl`) and model weights (LLM/NLI/encoder) are git-ignored. The pilot's parsed profiles/rankings and summaries are committed as evidence. See DATA_STATEMENT.md.
