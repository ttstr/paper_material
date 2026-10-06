"""Unified trainer for all non-LLM baselines (one evaluator for all methods).

Models (``model:`` key of the model yaml):
  sasrec   - SASRec, full-catalogue CE at every non-pad position (Stage-0 / reference model)
  sasrec_t - SASRec + frozen sentence-embedding item text features (same backbone)
  gru4rec  - GRU4Rec, full-catalogue CE at every non-pad position
  bert4rec - BERT4Rec, Cloze/masked-item CE; eval appends [MASK]
  mf_bpr   - MF-BPR (non-sequential reference), BPR with uniform negatives
Early stopping on valid NDCG@10; best checkpoint re-evaluated on valid+test with
``pcdrec.evaluator.evaluate_split``; per-user ranks saved to results/per_user/<tag>.npz.
"""

from __future__ import annotations

import argparse
import csv
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

from pcdrec.data.load_splits import load_data_config, load_processed, load_and_process, load_yaml, subset_users
from pcdrec.evaluator import evaluate_split, pad_history  # noqa: F401  (re-exported for back-compat)
from pcdrec.models.sasrec import SASRec
from pcdrec.paths import REPO_ROOT, resolve


def rel(p) -> str:
    """Repo-relative string for logging (never write machine-specific absolute paths)."""
    p = Path(p)
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


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


class SeqTrainDataset(Dataset):
    """BERT4Rec: last max_len train items per user, left-padded (masking happens per batch)."""

    def __init__(self, train_seq: dict, max_len: int, pad_id: int):
        self.rows = []
        for uid, events in train_seq.items():
            items = [e["item"] for e in events][-max_len:]
            self.rows.append([pad_id] * (max_len - len(items)) + items)
        self.rows = torch.tensor(self.rows, dtype=torch.long)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class MFTrainDataset(Dataset):
    """(user, positive item) pairs from train; negatives sampled per batch in the loop."""

    def __init__(self, train_seq: dict):
        pairs = [(u, e["item"]) for u, evs in train_seq.items() for e in evs]
        self.pairs = torch.tensor(pairs, dtype=torch.long)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i, 0], self.pairs[i, 1]


def sample_negatives(users: torch.Tensor, n_items: int, user_items: dict, rng: np.random.Generator) -> torch.Tensor:
    neg = rng.integers(0, n_items, size=len(users))
    for k, u in enumerate(users.tolist()):
        seen = user_items[u]
        while int(neg[k]) in seen:
            neg[k] = rng.integers(0, n_items)
    return torch.as_tensor(neg, dtype=torch.long)


def build_model(cfg: dict, n_items: int, n_users: int, max_len: int, data_cfg: dict):
    name = cfg.get("model", "sasrec")
    common = dict(
        hidden_size=int(cfg["hidden_size"]),
        initializer_range=float(cfg.get("initializer_range", 0.02)),
    )
    if name in ("sasrec", "sasrec_t"):
        kw = dict(
            n_items=n_items,
            n_layers=int(cfg["n_layers"]),
            n_heads=int(cfg["n_heads"]),
            inner_size=int(cfg.get("inner_size", cfg["hidden_size"] * 4)),
            max_seq_length=max_len,
            hidden_dropout_prob=float(cfg["hidden_dropout_prob"]),
            attn_dropout_prob=float(cfg["attn_dropout_prob"]),
            **common,
        )
        if name == "sasrec":
            return SASRec(**kw)
        from pcdrec.encode.text_encoder import load_or_build
        from pcdrec.models.sasrec_t import SASRecT

        emb = load_or_build(data_cfg["processed_dir"], cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"))
        return SASRecT(text_emb=emb, **kw)
    if name == "gru4rec":
        from pcdrec.models.gru4rec import GRU4Rec

        return GRU4Rec(n_items=n_items, gru_hidden=int(cfg.get("gru_hidden", cfg["hidden_size"])),
                       n_layers=int(cfg.get("n_layers", 1)), dropout=float(cfg["hidden_dropout_prob"]),
                       max_seq_length=max_len, **common)
    if name == "bert4rec":
        from pcdrec.models.bert4rec import BERT4Rec

        return BERT4Rec(n_items=n_items, n_layers=int(cfg["n_layers"]), n_heads=int(cfg["n_heads"]),
                        inner_size=int(cfg.get("inner_size", cfg["hidden_size"] * 4)), max_seq_length=max_len,
                        hidden_dropout_prob=float(cfg["hidden_dropout_prob"]),
                        attn_dropout_prob=float(cfg["attn_dropout_prob"]),
                        mask_ratio=float(cfg.get("mask_ratio", 0.2)), **common)
    if name == "mf_bpr":
        from pcdrec.models.mf_bpr import MFBPR

        return MFBPR(n_users=n_users, n_items=n_items, hidden_size=int(cfg["hidden_size"]),
                     initializer_range=float(cfg.get("initializer_range", 0.1)))
    raise ValueError(f"unknown model {name}")


def parse_sets(sets: list[str] | None) -> dict:
    """--set a.b=1 --set c=0.5 -> nested dict (values parsed as yaml)."""
    out: dict = {}
    for item in sets or []:
        k, v = item.split("=", 1)
        cur = out
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = yaml.safe_load(v)
    return out


def train_main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Train a non-LLM baseline (SASRec / SASRec-T / GRU4Rec / BERT4Rec / MF-BPR)")
    ap.add_argument("--model-config", default=str(REPO_ROOT / "configs/model/sasrec.yaml"))
    ap.add_argument("--data-config", default=str(REPO_ROOT / "configs/data/beauty.yaml"))
    ap.add_argument("--max-users", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--patience", type=int, default=None, help="Early-stop patience on valid NDCG@10")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--threads", type=int, default=None, help="torch.set_num_threads (CPU)")
    ap.add_argument("--set", action="append", default=None, help="override config key, e.g. --set hidden_dropout_prob=0.5")
    ap.add_argument("--skip-process", action="store_true", help="Use existing processed pickle")
    args = ap.parse_args(argv)

    cfg = load_merged_config(args.model_config, args.data_config, parse_sets(args.set))
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
    if args.patience is not None:
        cfg["optim"]["patience"] = args.patience
    if args.lr is not None:
        cfg["optim"]["lr"] = args.lr
    if args.threads is not None:
        cfg["threads"] = args.threads
    if cfg.get("threads"):
        torch.set_num_threads(int(cfg["threads"]))
    model_name = cfg.get("model", "sasrec")

    set_seed(int(cfg["seed"]))
    device = resolve_device(str(cfg.get("device", "auto")))

    data_cfg_path = cfg.get("data_cfg_path", args.data_config)
    data_cfg = load_data_config(data_cfg_path)
    processed_dir = Path(data_cfg["processed_dir"])
    pkl = processed_dir / "beauty_loo.pkl"
    if not args.skip_process or not pkl.exists():
        load_and_process(data_cfg_path, write_processed=True)
    bundle = load_processed(processed_dir)
    bundle = subset_users(bundle, cfg.get("max_users"), seed=int(cfg["seed"]))

    n_items = int(bundle["n_items"])
    n_users_total = int(max(bundle["train_seq"].keys())) + 1
    max_len = int(cfg.get("max_seq_length", data_cfg.get("max_len", 50)))
    model = build_model(cfg, n_items, n_users_total, max_len, data_cfg).to(device)

    batch_size = int(cfg["optim"]["batch_size"])
    if model_name == "mf_bpr":
        train_ds = MFTrainDataset(bundle["train_seq"])
        user_items = {u: {e["item"] for e in evs} for u, evs in bundle["train_seq"].items()}
        neg_rng = np.random.default_rng(int(cfg["seed"]))
    elif model_name == "bert4rec":
        train_ds = SeqTrainDataset(bundle["train_seq"], max_len=max_len, pad_id=model.pad_id)
    else:
        train_ds = SASRecTrainDataset(bundle["train_seq"], max_len=max_len, pad_id=model.pad_id)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                        num_workers=int(cfg["optim"].get("num_workers", 0)))
    optim = torch.optim.Adam(model.parameters(), lr=float(cfg["optim"]["lr"]),
                             weight_decay=float(cfg["optim"].get("weight_decay", 0.0)))

    ckpt_dir = resolve(cfg.get("checkpoint_dir", "results/checkpoints"))
    results_dir = resolve(cfg.get("results_dir", "results"))
    per_user_dir = results_dir / "per_user"
    for d in (ckpt_dir, results_dir, per_user_dir):
        d.mkdir(parents=True, exist_ok=True)

    tag = args.tag or model_name
    if bundle.get("subset"):
        tag = f"{tag}_subset{bundle['subset']['n_users']}"

    best_metric = -1.0
    best_path = ckpt_dir / f"{tag}_best.pt"
    patience = int(cfg["optim"]["patience"])
    bad = 0
    history = []
    t0 = time.time()
    eval_bs = min(128, batch_size)

    epochs = int(cfg["optim"]["epochs"])
    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        model.train()
        losses = []
        for batch in tqdm(loader, desc=f"epoch {epoch}", leave=False):
            if model_name == "mf_bpr":
                u, i = batch
                j = sample_negatives(u, n_items, user_items, neg_rng)
                loss = model.bpr_loss(u.to(device), i.to(device), j.to(device), l2=float(cfg.get("l2", 0.0)))
            elif model_name == "bert4rec":
                # Cloze objective on the last max_len train items (input = seq_in plus its final target)
                inp, tgt = model.mask_batch(batch)
                inp, tgt = inp.to(device), tgt.to(device)
                h = model(inp)
                m = tgt.ne(-100)
                logits = model.head(h[m])
                loss = F.cross_entropy(logits, tgt[m])
            else:
                seq_in, seq_tgt = batch
                seq_in = seq_in.to(device)
                seq_tgt = seq_tgt.to(device)
                # Only compute full-catalog logits at non-ignored (non-pad) positions.
                # Identical to CE(ignore_index=-100, reduction=mean) over [B*L].
                h = model(seq_in)  # [B, L, H]
                valid_pos = seq_tgt.ne(-100)
                logits = h[valid_pos] @ model.output_item_weights().t()  # [M, n_items]
                loss = F.cross_entropy(logits, seq_tgt[valid_pos])
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())

        t_train = time.time() - t_ep
        valid_metrics = evaluate_split(model, bundle["train_seq"], bundle["valid_target"], max_len=max_len,
                                       device=device, ks=(5, 10, 20), batch_size=eval_bs, history_mode="train")
        t_epoch = time.time() - t_ep
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else None,
            "train_sec": t_train,
            "epoch_sec": t_epoch,
            **{f"valid_{k}": v for k, v in valid_metrics.items()},
        }
        history.append(row)
        metric = valid_metrics["ndcg@10"]
        print(
            f"epoch={epoch} loss={row['train_loss']:.4f} "
            f"valid_ndcg@10={metric:.6f} valid_hr@10={valid_metrics['hr@10']:.6f} "
            f"train_sec={t_train:.1f} epoch_sec={t_epoch:.1f} total_sec={time.time() - t0:.0f}",
            flush=True,
        )

        if metric > best_metric:
            best_metric = metric
            bad = 0
            torch.save({"model_state": model.state_dict(), "cfg": cfg, "n_items": n_items, "max_len": max_len,
                        "epoch": epoch, "valid_metrics": valid_metrics, "subset": bundle.get("subset"),
                        "tag": tag, "model": model_name}, best_path)
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop at epoch {epoch} (patience={patience})", flush=True)
                break

    elapsed = time.time() - t0
    epochs_run = len(history)
    stopped_early = epochs_run < epochs

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    valid_metrics, users_v, ranks_v = evaluate_split(
        model, bundle["train_seq"], bundle["valid_target"], max_len, device, history_mode="train", return_ranks=True)
    test_metrics, users_t, ranks_t = evaluate_split(
        model, bundle["train_seq"], bundle["test_target"], max_len, device, history_mode="train_valid",
        valid_target=bundle["valid_target"], return_ranks=True)
    assert np.array_equal(users_v, users_t)
    np.savez_compressed(per_user_dir / f"{tag}.npz", users=users_t.astype(np.int32),
                        valid_rank=ranks_v.astype(np.int32), test_rank=ranks_t.astype(np.int32))

    cfg_out = {
        "model_name": model_name,
        "model": rel(args.model_config),
        "data": rel(args.data_config),
        "max_users": cfg.get("max_users"),
        "epochs": epochs,
        "batch_size": cfg["optim"]["batch_size"],
        "n_layers": cfg.get("n_layers"),
        "n_heads": cfg.get("n_heads"),
        "hidden_size": cfg["hidden_size"],
        "max_seq_length": max_len,
        "inner_size": int(cfg.get("inner_size", cfg["hidden_size"] * 4)),
        "lr": float(cfg["optim"]["lr"]),
        "weight_decay": float(cfg["optim"].get("weight_decay", 0.0)),
        "patience": patience,
        "early_stop_metric": "valid ndcg@10",
        "hidden_dropout_prob": cfg.get("hidden_dropout_prob"),
        "attn_dropout_prob": cfg.get("attn_dropout_prob"),
        "threads": torch.get_num_threads(),
        "overrides": parse_sets(args.set),
        "eval": "full-catalog ranking, history items NOT filtered; valid uses train history, test uses train+valid history",
    }
    for k in ("mask_ratio", "gru_hidden", "text_encoder", "l2"):
        if k in cfg:
            cfg_out[k] = cfg[k]
    out = {
        "tag": tag,
        "model": model_name,
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
        "epochs_run": epochs_run,
        "stopped_early": stopped_early,
        "mean_epoch_sec": float(np.mean([r["epoch_sec"] for r in history])) if history else None,
        "torch_threads": torch.get_num_threads(),
        "checkpoint": rel(best_path),
        "per_user_ranks": rel(per_user_dir / f"{tag}.npz"),
        "seed": cfg["seed"],
        "config": cfg_out,
        "history": history,
        "note": "Metrics written by train.py; do not hand-edit.",
    }

    metrics_path = results_dir / f"{tag}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    flat = {
        "tag": tag, "model": model_name, "n_users": bundle["n_users"], "subset": bool(bundle.get("subset")),
        "seed": cfg["seed"], "best_epoch": ckpt["epoch"], "device": str(device),
        **{f"valid_{k}": valid_metrics[k] for k in ("hr@5", "hr@10", "hr@20", "ndcg@5", "ndcg@10", "ndcg@20")},
        **{f"test_{k}": test_metrics[k] for k in ("hr@5", "hr@10", "hr@20", "ndcg@5", "ndcg@10", "ndcg@20")},
        "elapsed_sec": elapsed, "checkpoint": rel(best_path),
    }
    with open(results_dir / f"{tag}_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(flat.keys()))
        w.writeheader()
        w.writerow(flat)

    print(json.dumps({k: out[k] for k in ("tag", "valid", "test", "checkpoint")}, indent=2))
    print(f"wrote {rel(metrics_path)}")
    return out


if __name__ == "__main__":
    train_main()
