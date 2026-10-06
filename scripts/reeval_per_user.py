#!/usr/bin/env python
"""Re-evaluate existing checkpoints with the shared evaluator and save per-user ranks.

Used for runs trained before per-user ranks were saved (Stage-0 SASRec seeds).
Asserts that the re-evaluated valid/test metrics equal those stored in the
run's metrics JSON (same evaluator, same protocol), then writes
results/per_user/<tag>.npz. Usage:
  python scripts/reeval_per_user.py results/sasrec_full_s4{2,3,4,5,6}_metrics.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from pcdrec.data.load_splits import load_data_config, load_processed
from pcdrec.evaluator import evaluate_split
from pcdrec.paths import REPO_ROOT, resolve
from pcdrec.train import build_model


def main(paths: list[str]) -> None:
    torch.set_num_threads(4)
    data_cfg = load_data_config(REPO_ROOT / "configs/data/beauty.yaml")
    bundle = load_processed(data_cfg["processed_dir"])
    out_dir = REPO_ROOT / "results/per_user"
    out_dir.mkdir(parents=True, exist_ok=True)
    for mp in paths:
        d = json.loads(Path(mp).read_text())
        ck_path = resolve(d["checkpoint"]) if not Path(d["checkpoint"]).exists() else Path(d["checkpoint"])
        if not ck_path.exists():
            ck_path = REPO_ROOT / "results/checkpoints" / Path(d["checkpoint"]).name
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        cfg = dict(ck["cfg"])
        cfg.setdefault("model", "sasrec")
        model = build_model(cfg, int(ck["n_items"]), int(bundle["n_users"]), int(ck["max_len"]), data_cfg)
        model.load_state_dict(ck["model_state"])
        L = int(ck["max_len"])
        v, uv, rv = evaluate_split(model, bundle["train_seq"], bundle["valid_target"], L, torch.device("cpu"),
                                   history_mode="train", return_ranks=True)
        t, ut, rt = evaluate_split(model, bundle["train_seq"], bundle["test_target"], L, torch.device("cpu"),
                                   history_mode="train_valid", valid_target=bundle["valid_target"], return_ranks=True)
        for split, new in (("valid", v), ("test", t)):
            for k in ("hr@10", "ndcg@10", "ndcg@20", "mean_rank"):
                assert math.isclose(new[k], d[split][k], rel_tol=1e-9, abs_tol=1e-12), (mp, split, k, new[k], d[split][k])
        np.savez_compressed(out_dir / f"{d['tag']}.npz", users=ut.astype(np.int32),
                            valid_rank=rv.astype(np.int32), test_rank=rt.astype(np.int32))
        print(f"{d['tag']}: re-eval matches stored metrics (test ndcg@10={t['ndcg@10']:.6f}); wrote per_user/{d['tag']}.npz")


if __name__ == "__main__":
    main(sys.argv[1:])
