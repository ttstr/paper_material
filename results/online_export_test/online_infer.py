"""Online SASRec inference (exported). No LLM dependencies."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from pcdrec.models.sasrec import SASRec


def load_online(export_dir: str | Path, device: str = "cpu") -> SASRec:
    export_dir = Path(export_dir)
    with open(export_dir / "online_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    ckpt = torch.load(export_dir / "sasrec.pt", map_location=device, weights_only=False)
    model = SASRec(
        n_items=int(meta["n_items"]),
        hidden_size=int(meta["hidden_size"]),
        n_layers=int(meta["n_layers"]),
        n_heads=int(meta["n_heads"]),
        inner_size=int(meta["inner_size"]),
        max_seq_length=int(meta["max_seq_length"]),
        hidden_dropout_prob=float(meta.get("hidden_dropout_prob", 0.0)),
        attn_dropout_prob=float(meta.get("attn_dropout_prob", 0.0)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def recommend(model: SASRec, item_ids: list[int], topk: int = 10) -> list[int]:
    """item_ids: recent interaction ids (oldest -> newest). Returns top-k item ids."""
    max_len = model.max_seq_length
    pad = model.pad_id
    seq = item_ids[-max_len:]
    padded = [pad] * (max_len - len(seq)) + seq
    x = torch.tensor([padded], dtype=torch.long, device=next(model.parameters()).device)
    logits = model.predict_logits(x)[0]
    for i in set(item_ids):
        if 0 <= i < model.n_items:
            logits[i] = -1e9
    return torch.topk(logits, k=min(topk, model.n_items)).indices.tolist()
