"""Train Stage-0 SASRec with full-catalog CE; early-stop on valid NDCG@10."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# allow `python -m pcdrec.train` from repo with PYTHONPATH=src
from pcdrec.data.load_splits import load_processed, load_and_process, load_yaml, subset_users
from pcdrec.metrics.ranking import evaluate_ranks, ranks_from_scores
from pcdrec.models.sasrec import SASRec


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


class SASRecTrainDataset(Dataset):
    """Each sample: left-padded input seq -> next-item targets at each position."""

    def __init__(self, train_seq: dict, max_len: int, pad_id: int):
        self.samples: list[tuple[list[int], list[int]]] = []
        self.max_len = max_len
        self.pad_id = pad_id
        for uid, events in train_seq.items():
            items = [e["item"] for e in events]
            if len(items) < 2:
                continue
            # use full train sequence: predict items[1:] from items[:-1]
            seq = items
            # sliding: we pack last max_len+1 items
            if len(seq) > max_len + 1:
                seq = seq[-(max_len + 1) :]
            inp = seq[:-1]
            tgt = seq[1:]
            self.samples.append((inp, tgt))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        inp, tgt = self.samples[idx]
        L = self.max_len
        inp_p = [self.pad_id] * (L - len(inp)) + inp
        tgt_p = [-100] * (L - len(tgt)) + tgt  # ignore_index
        return (
            torch.tensor(inp_p, dtype=torch.long),
            torch.tensor(tgt_p, dtype=torch.long),
        )


def pad_history(items: list[int], max_len: int, pad_id: int) -> torch.Tensor:
    if len(items) > max_len:
        items = items[-max_len:]
    padded = [pad_id] * (max_len - len(items)) + items
    return torch.tensor(padded, dtype=torch.long)


@torch.no_grad()
def evaluate_split(
    model: SASRec,
    train_seq: dict,
    targets: dict,
    max_len: int,
    device: torch.device,
    ks=(5, 10, 20),
    batch_size: int = 64,
    history_mode: str = "train",  # train | train_valid
    valid_target: dict | None = None,
) -> dict:
    model.eval()
    users = sorted(targets.keys())
    ranks: list[int] = []
    pad_id = model.pad_id

    for start in range(0, len(users), batch_size):
        batch_users = users[start : start + batch_size]
        seqs = []
        tgts = []
        for u in batch_users:
            hist = [e["item"] for e in train_seq[u]]
            if history_mode == "train_valid" and valid_target is not None:
                hist = hist + [valid_target[u]["item"]]
            seqs.append(pad_history(hist, max_len, pad_id))
            tgts.append(targets[u]["item"])
        seq_t = torch.stack(seqs, dim=0).to(device)
        logits = model.predict_logits(seq_t)  # [B, n_items]
        # mask history items? Standard SASRec full-rank often does NOT mask; keep unmasked for LOO fairness.
        scores = logits.cpu().numpy()
        for i, t in enumerate(tgts):
            ranks.append(ranks_from_scores(scores[i], int(t)))

    return evaluate_ranks(ranks, ks)


def deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_merged_config(model_cfg: str, data_cfg: str | None, overrides: dict | None = None) -> dict:
    m = load_yaml(model_cfg)
    if data_cfg:
        d = load_yaml(data_cfg)
        m = deep_merge({"data": d}, m)
        m["data_cfg_path"] = data_cfg
    m["model_cfg_path"] = model_cfg
    if overrides:
        m = deep_merge(m, overrides)
    return m


def train_main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Train SASRec CE (Stage-0)")
    ap.add_argument("--model-config", default="/workspace/pcdrec/configs/model/sasrec.yaml")
    ap.add_argument("--data-config", default="/workspace/pcdrec/configs/data/beauty.yaml")
    ap.add_argument("--max-users", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default="sasrec")
    ap.add_argument("--skip-process", action="store_true", help="Use existing processed pickle")
    args = ap.parse_args(argv)

    cfg = load_merged_config(args.model_config, args.data_config)
    if args.max_users is not None:
        cfg["max_users"] = args.max_users
    if args.epochs is not None:
        cfg["optim"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["optim"]["batch_size"] = args.batch_size
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device is not None:
        cfg["device"] = args.device

    set_seed(int(cfg["seed"]))
    device = resolve_device(str(cfg.get("device", "auto")))

    data_cfg_path = cfg.get("data_cfg_path", args.data_config)
    data_cfg = load_yaml(data_cfg_path)
    processed_dir = Path(data_cfg["processed_dir"])
    pkl = processed_dir / "beauty_loo.pkl"
    if not args.skip_process or not pkl.exists():
        load_and_process(data_cfg_path, write_processed=True)
    bundle = load_processed(processed_dir)
    bundle = subset_users(bundle, cfg.get("max_users"), seed=int(cfg["seed"]))

    n_items = int(bundle["n_items"])
    max_len = int(cfg.get("max_seq_length", data_cfg.get("max_len", 50)))
    model = SASRec(
        n_items=n_items,
        hidden_size=int(cfg["hidden_size"]),
        n_layers=int(cfg["n_layers"]),
        n_heads=int(cfg["n_heads"]),
        inner_size=int(cfg.get("inner_size", cfg["hidden_size"] * 4)),
        max_seq_length=max_len,
        hidden_dropout_prob=float(cfg["hidden_dropout_prob"]),
        attn_dropout_prob=float(cfg["attn_dropout_prob"]),
        initializer_range=float(cfg.get("initializer_range", 0.02)),
    ).to(device)

    train_ds = SASRecTrainDataset(bundle["train_seq"], max_len=max_len, pad_id=model.pad_id)
    loader = DataLoader(
        train_ds,
        batch_size=int(cfg["optim"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["optim"].get("num_workers", 0)),
    )
    optim = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["optim"]["lr"]),
        weight_decay=float(cfg["optim"].get("weight_decay", 0.0)),
    )

    ckpt_dir = Path(cfg.get("checkpoint_dir", "/workspace/pcdrec/results/checkpoints"))
    results_dir = Path(cfg.get("results_dir", "/workspace/pcdrec/results"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    tag = args.tag
    if bundle.get("subset"):
        tag = f"{tag}_subset{bundle['subset']['n_users']}"

    best_metric = -1.0
    best_path = ckpt_dir / f"{tag}_best.pt"
    patience = int(cfg["optim"]["patience"])
    bad = 0
    history = []
    t0 = time.time()

    epochs = int(cfg["optim"]["epochs"])
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for seq_in, seq_tgt in tqdm(loader, desc=f"epoch {epoch}", leave=False):
            seq_in = seq_in.to(device)
            seq_tgt = seq_tgt.to(device)
            logits = model.train_step_logits(seq_in)  # [B, L, n_items]
            B, L, N = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * L, N),
                seq_tgt.reshape(B * L),
                ignore_index=-100,
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())

        valid_metrics = evaluate_split(
            model,
            bundle["train_seq"],
            bundle["valid_target"],
            max_len=max_len,
            device=device,
            ks=(5, 10, 20),
            batch_size=min(128, int(cfg["optim"]["batch_size"])),
            history_mode="train",
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else None,
            **{f"valid_{k}": v for k, v in valid_metrics.items()},
        }
        history.append(row)
        metric = valid_metrics["ndcg@10"]
        print(
            f"epoch={epoch} loss={row['train_loss']:.4f} "
            f"valid_ndcg@10={metric:.6f} valid_hr@10={valid_metrics['hr@10']:.6f}",
            flush=True,
        )

        if metric > best_metric:
            best_metric = metric
            bad = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "cfg": cfg,
                    "n_items": n_items,
                    "max_len": max_len,
                    "epoch": epoch,
                    "valid_metrics": valid_metrics,
                    "subset": bundle.get("subset"),
                    "tag": tag,
                },
                best_path,
            )
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop at epoch {epoch} (patience={patience})", flush=True)
                break

    elapsed = time.time() - t0

    # reload best and evaluate valid+test
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    valid_metrics = evaluate_split(
        model, bundle["train_seq"], bundle["valid_target"], max_len, device, history_mode="train"
    )
    test_metrics = evaluate_split(
        model,
        bundle["train_seq"],
        bundle["test_target"],
        max_len,
        device,
        history_mode="train_valid",
        valid_target=bundle["valid_target"],
    )

    out = {
        "tag": tag,
        "subset": bundle.get("subset"),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "n_users": bundle["n_users"],
        "n_items": n_items,
        "n_params": model.num_parameters(),
        "best_epoch": ckpt["epoch"],
        "best_valid_ndcg@10": best_metric,
        "valid": valid_metrics,
        "test": test_metrics,
        "elapsed_sec": elapsed,
        "checkpoint": str(best_path),
        "seed": cfg["seed"],
        "config": {
            "model": args.model_config,
            "data": args.data_config,
            "max_users": cfg.get("max_users"),
            "epochs": epochs,
            "batch_size": cfg["optim"]["batch_size"],
            "n_layers": cfg["n_layers"],
            "n_heads": cfg["n_heads"],
            "hidden_size": cfg["hidden_size"],
            "max_seq_length": max_len,
        },
        "history": history,
        "note": "Metrics written by train.py; do not hand-edit.",
    }

    metrics_path = results_dir / f"{tag}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # also a slim csv row
    csv_path = results_dir / f"{tag}_metrics.csv"
    import csv

    flat = {
        "tag": tag,
        "n_users": bundle["n_users"],
        "subset": bool(bundle.get("subset")),
        "seed": cfg["seed"],
        "best_epoch": ckpt["epoch"],
        "device": str(device),
        **{f"valid_{k}": valid_metrics[k] for k in ("hr@5", "hr@10", "hr@20", "ndcg@5", "ndcg@10", "ndcg@20")},
        **{f"test_{k}": test_metrics[k] for k in ("hr@5", "hr@10", "hr@20", "ndcg@5", "ndcg@10", "ndcg@20")},
        "elapsed_sec": elapsed,
        "checkpoint": str(best_path),
    }
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(flat.keys()))
        w.writeheader()
        w.writerow(flat)

    print(json.dumps({k: out[k] for k in ("tag", "valid", "test", "checkpoint")}, indent=2))
    print(f"wrote {metrics_path}")
    return out


if __name__ == "__main__":
    train_main()
