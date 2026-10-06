#!/usr/bin/env python
"""Check an exported online model: (1) it loads as a plain SASRec without any LLM library, (2) the shared evaluator
reproduces the student's full test metrics from the export alone, (3) per-request CPU latency (batch 1).
Writes <out>/online_export_check.json.

  python scripts/check_online_export.py --export-dir results/pilot/online_export \
      --student-metrics results/pilot/pcdrec_pilot_s42_metrics.json --out results/pilot
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from pcdrec.data.load_splits import load_data_config, load_processed
from pcdrec.evaluator import evaluate_split, pad_history
from pcdrec.paths import REPO_ROOT, resolve

LLM_MODULES = ("vllm", "openai", "transformers", "sentence_transformers", "pcdrec.llm_offline", "pcdrec.verify")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--student-metrics", default=None)
    ap.add_argument("--out", default="results/pilot")
    ap.add_argument("--n-latency", type=int, default=2000)
    args = ap.parse_args()
    torch.set_num_threads(1)
    exp = resolve(args.export_dir)
    spec = importlib.util.spec_from_file_location("online_infer", exp / "online_infer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.load_online(exp)
    meta = json.loads((exp / "online_meta.json").read_text())
    data_cfg = load_data_config(str(REPO_ROOT / "configs/data/beauty.yaml"))
    b = load_processed(data_cfg["processed_dir"])
    max_len = int(meta["max_seq_length"])
    torch.set_num_threads(4)
    tm = evaluate_split(model, b["train_seq"], b["test_target"], max_len, torch.device("cpu"),
                        history_mode="train_valid", valid_target=b["valid_target"])
    torch.set_num_threads(1)
    users = sorted(b["test_target"])[: args.n_latency]
    lat = []
    with torch.no_grad():
        for u in users:
            hist = [e["item"] for e in b["train_seq"][u]] + [b["valid_target"][u]["item"]]
            x = pad_history(hist, max_len, model.pad_id).unsqueeze(0)
            t0 = time.perf_counter()
            s = model.score(None, x)
            torch.topk(s[0], 10)
            lat.append((time.perf_counter() - t0) * 1000)
    loaded_llm = sorted(m for m in sys.modules if any(m == x or m.startswith(x + ".") for x in LLM_MODULES))
    res = {"export_dir": str(exp.relative_to(REPO_ROOT)) if exp.is_relative_to(REPO_ROOT) else exp.name, "online_llm_calls": meta["online_llm_calls"],
           "state_dict_keys": len(torch.load(exp / "sasrec.pt", weights_only=False)["model_state"]),
           "n_params": meta["n_params"], "dropped_training_only_modules": meta.get("dropped_training_only_keys"),
           "llm_modules_imported_during_check": loaded_llm,
           "export_test_ndcg@10": tm["ndcg@10"], "export_test_hr@10": tm["hr@10"],
           "latency_ms_batch1_1thread_p50": float(np.percentile(lat, 50)),
           "latency_ms_batch1_1thread_p95": float(np.percentile(lat, 95)), "latency_n_requests": len(lat)}
    if args.student_metrics:
        sm = json.loads(resolve(args.student_metrics).read_text())
        res["student_test_ndcg@10"] = sm["test"]["ndcg@10"]
        res["export_matches_student"] = bool(abs(sm["test"]["ndcg@10"] - tm["ndcg@10"]) < 1e-9
                                             and abs(sm["test"]["hr@10"] - tm["hr@10"]) < 1e-9)
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "online_export_check.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
