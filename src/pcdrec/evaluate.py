"""Evaluate any baseline / student checkpoint with the shared full-catalogue evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pcdrec.data.load_splits import load_data_config, load_processed, subset_users
from pcdrec.evaluator import evaluate_split
from pcdrec.paths import REPO_ROOT
from pcdrec.train import build_model, resolve_device


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-config", default=str(REPO_ROOT / "configs/data/beauty.yaml"))
    ap.add_argument("--split", choices=["valid", "test", "both"], default="both")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None, help="Optional json path under results/")
    ap.add_argument("--max-users", type=int, default=None)
    args = ap.parse_args(argv)

    device = resolve_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = dict(ckpt.get("cfg", {}))
    cfg.setdefault("model", ckpt.get("model", "sasrec"))
    n_items = int(ckpt["n_items"])
    max_len = int(ckpt["max_len"])
    data_cfg = load_data_config(args.data_config)
    bundle = load_processed(data_cfg["processed_dir"])
    model = build_model(cfg, n_items, int(max(bundle["train_seq"]) + 1), max_len, data_cfg).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    max_users = args.max_users
    if max_users is None and ckpt.get("subset"):
        max_users = ckpt["subset"]["n_users"]
        bundle = subset_users(bundle, max_users, seed=ckpt["subset"].get("seed", cfg.get("seed", 42)))
    elif max_users is not None:
        bundle = subset_users(bundle, max_users, seed=int(cfg.get("seed", 42)))

    out = {"checkpoint": str(args.checkpoint), "subset": bundle.get("subset") or ckpt.get("subset"),
           "n_users": bundle["n_users"], "device": str(device)}
    if args.split in ("valid", "both"):
        out["valid"] = evaluate_split(model, bundle["train_seq"], bundle["valid_target"], max_len, device,
                                      history_mode="train")
    if args.split in ("test", "both"):
        out["test"] = evaluate_split(model, bundle["train_seq"], bundle["test_target"], max_len, device,
                                     history_mode="train_valid", valid_target=bundle["valid_target"])
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {out_path}")
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
