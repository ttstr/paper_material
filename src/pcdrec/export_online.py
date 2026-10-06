"""Export online inference module: SASRec weights only (no LLM / no auxiliary heads).

Stage-0 stub: copies checkpoint into an online-ready package and writes a
pure-SASRec inference helper that must not import vllm/transformers generative APIs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pcdrec.models.sasrec import SASRec


ONLINE_INFER_PY = '''"""Online SASRec inference (exported). No LLM dependencies."""

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
'''


def export(checkpoint: str, out_dir: str) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", {})

    state = ckpt["model_state"]
    torch.save({"model_state": state}, out / "sasrec.pt")

    meta = {
        "n_items": int(ckpt["n_items"]),
        "max_seq_length": int(ckpt["max_len"]),
        "hidden_size": int(cfg.get("hidden_size", 64)),
        "n_layers": int(cfg.get("n_layers", 2)),
        "n_heads": int(cfg.get("n_heads", 2)),
        "inner_size": int(cfg.get("inner_size", 256)),
        "hidden_dropout_prob": 0.0,
        "attn_dropout_prob": 0.0,
        "source_checkpoint": str(checkpoint),
        "online_llm_calls": 0,
        "has_aux_heads": False,
        "note": "SASRec-only online export (Stage-0 stub).",
    }

    model = SASRec(
        n_items=meta["n_items"],
        hidden_size=meta["hidden_size"],
        n_layers=meta["n_layers"],
        n_heads=meta["n_heads"],
        inner_size=meta["inner_size"],
        max_seq_length=meta["max_seq_length"],
    )
    model.load_state_dict(state)
    meta["n_params"] = model.num_parameters()

    with open(out / "online_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    (out / "online_infer.py").write_text(ONLINE_INFER_PY, encoding="utf-8")
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", default="results/online_export")
    args = ap.parse_args(argv)
    meta = export(args.checkpoint, args.out_dir)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
