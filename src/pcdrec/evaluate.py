"""Evaluate a SASRec checkpoint with full-catalog ranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pcdrec.data.load_splits import load_processed, load_yaml, subset_users
from pcdrec.models.sasrec import SASRec
from pcdrec.train import evaluate_split, resolve_device, set_seed


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-config", default="/workspace/pcdrec/configs/data/beauty.yaml")
    ap.add_argument("--split", choices=["valid", "test", "both"], default="both")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None, help="Optional json path under results/")
    ap.add_argument("--max-users", type=int, default=None)
    args = ap.parse_args(argv)

    device = resolve_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", {})
    n_items = int(ckpt["n_items"])
    max_len = int(ckpt["max_len"])

    model = SASRec(
        n_items=n_items,
        hidden_size=int(cfg.get("hidden_size", 64)),
        n_layers=int(cfg.get("n_layers", 2)),
        n_heads=int(cfg.get("n_heads", 2)),
        inner_size=int(cfg.get("inner_size", 256)),
        max_seq_length=max_len,
        hidden_dropout_prob=float(cfg.get("hidden_dropout_prob", 0.2)),
        attn_dropout_prob=float(cfg.get("attn_dropout_prob", 0.2)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    data_cfg = load_yaml(args.data_config)
    bundle = load_processed(data_cfg["processed_dir"])
    max_users = args.max_users
    if max_users is None and ckpt.get("subset"):
        max_users = ckpt["subset"]["n_users"]
        seed = ckpt["subset"].get("seed", cfg.get("seed", 42))
        bundle = subset_users(bundle, max_users, seed=seed)
    elif max_users is not None:
        bundle = subset_users(bundle, max_users, seed=int(cfg.get("seed", 42)))

    out = {
        "checkpoint": str(args.checkpoint),
        "subset": bundle.get("subset") or ckpt.get("subset"),
        "n_users": bundle["n_users"],
        "device": str(device),
    }
    if args.split in ("valid", "both"):
        out["valid"] = evaluate_split(
            model, bundle["train_seq"], bundle["valid_target"], max_len, device, history_mode="train"
        )
    if args.split in ("test", "both"):
        out["test"] = evaluate_split(
            model,
            bundle["train_seq"],
            bundle["test_target"],
            max_len,
            device,
            history_mode="train_valid",
            valid_target=bundle["valid_target"],
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"wrote {out_path}")

    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
