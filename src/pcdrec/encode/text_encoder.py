"""Frozen sentence-embedding encoder for item text (shared by SASRec-T, PCS and L_pref).

Encodes every catalogue item once (title + categories + brand, the same
``item_text`` used everywhere else) and caches an L2-normalised float32 matrix
under ``<processed_dir>/text_emb/`` (gitignored). Row i = item id i.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from pcdrec.data.load_splits import load_data_config, load_processed
from pcdrec.paths import REPO_ROOT

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def cache_path(processed_dir: str | Path, model_name: str) -> Path:
    return Path(processed_dir) / "text_emb" / (model_name.replace("/", "__") + ".npy")


def item_texts(bundle: dict) -> list[str]:
    n = int(bundle["n_items"])
    return [bundle["item_text"].get(i, "") or "[empty]" for i in range(n)]


def encode_texts(texts: list[str], model_name: str = DEFAULT_MODEL, batch_size: int = 128,
                 max_seq_length: int = 128, device: str = "cpu") -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_seq_length
    emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                       show_progress_bar=False, convert_to_numpy=True)
    return emb.astype(np.float32)


_ENCODER_CACHE: dict = {}


def encode_sentences(texts: list[str], model_name: str = DEFAULT_MODEL, max_seq_length: int = 128) -> np.ndarray:
    """Encode arbitrary sentences (e.g. LLM claims) with a process-wide cached encoder."""
    from sentence_transformers import SentenceTransformer

    if model_name not in _ENCODER_CACHE:
        m = SentenceTransformer(model_name, device="cpu")
        m.max_seq_length = max_seq_length
        _ENCODER_CACHE[model_name] = m
    if not texts:
        return np.zeros((0, _ENCODER_CACHE[model_name].get_sentence_embedding_dimension()), np.float32)
    return _ENCODER_CACHE[model_name].encode(texts, batch_size=64, normalize_embeddings=True,
                                             show_progress_bar=False, convert_to_numpy=True).astype(np.float32)


def load_or_build(processed_dir: str | Path, model_name: str = DEFAULT_MODEL, bundle: dict | None = None,
                  max_seq_length: int = 128) -> np.ndarray:
    path = cache_path(processed_dir, model_name)
    if path.exists():
        return np.load(path)
    bundle = bundle or load_processed(processed_dir)
    texts = item_texts(bundle)
    t0 = time.time()
    emb = encode_texts(texts, model_name, max_seq_length=max_seq_length)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    info = {
        "model": model_name,
        "n_items": len(texts),
        "dim": int(emb.shape[1]),
        "max_seq_length": max_seq_length,
        "normalized": True,
        "text_fields": bundle.get("stats", {}).get("text_fields"),
        "texts_sha256": hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest(),
        "emb_sha256": hashlib.sha256(emb.tobytes()).hexdigest(),
        "encode_sec": time.time() - t0,
    }
    path.with_suffix(".json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    return emb


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", default=str(REPO_ROOT / "configs/data/beauty.yaml"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args(argv)
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)
    cfg = load_data_config(args.data_config)
    emb = load_or_build(cfg["processed_dir"], args.model)
    print(emb.shape)


if __name__ == "__main__":
    main()
