"""Load Amazon Beauty LOO splits, validate against manifest, build sequences, export processed."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .text_fields import DEFAULT_TEXT_FIELDS, item_text_table
from ..paths import REPO_ROOT, apply_data_env


INTERACTION_COLS = ["USER", "ITEM", "RATING", "TIMESTAMP"]
META_COLS = ["ITEM", "TITLE", "SALES_TYPE", "SALES_RANK", "CATEGORIES", "PRICE", "BRAND"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data_config(path: str | Path) -> dict:
    """Load a data yaml and resolve repo-relative paths + PCDREC_DATA_DIR overrides."""
    return apply_data_env(load_yaml(path))


def _read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def validate_against_manifest(
    raw_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Check sha256 and row counts; return per-file reports."""
    report: dict[str, Any] = {}
    for name, meta in manifest.items():
        path = raw_dir / name
        expected_sha = meta["sha256"]
        expected_rows = meta["rows"]
        actual_sha = sha256_file(path)
        df = _read_tsv(path)
        actual_rows = len(df)
        ok = (actual_sha == expected_sha) and (actual_rows == expected_rows)
        report[name] = {
            "path": str(path),
            "sha256_ok": actual_sha == expected_sha,
            "rows_ok": actual_rows == expected_rows,
            "actual_sha256": actual_sha,
            "actual_rows": actual_rows,
            "ok": ok,
        }
        if not ok:
            raise ValueError(
                f"Manifest validation failed for {name}: "
                f"sha_ok={actual_sha == expected_sha} rows={actual_rows}/{expected_rows}"
            )
    return report


def _check_split_invariants(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    meta: pd.DataFrame,
) -> dict[str, Any]:
    train_users = set(train["USER"].unique())
    valid_users = set(valid["USER"].unique())
    test_users = set(test["USER"].unique())

    if not (train_users == valid_users == test_users):
        raise AssertionError(
            f"User set mismatch: train={len(train_users)} valid={len(valid_users)} "
            f"test={len(test_users)} "
            f"train-valid={len(train_users - valid_users)} "
            f"valid-train={len(valid_users - train_users)}"
        )

    vc = valid.groupby("USER").size()
    tc = test.groupby("USER").size()
    if not (vc == 1).all():
        raise AssertionError(f"valid must have exactly 1 row per user; bad={int((vc != 1).sum())}")
    if not (tc == 1).all():
        raise AssertionError(f"test must have exactly 1 row per user; bad={int((tc != 1).sum())}")

    all_items = set(train["ITEM"]).union(valid["ITEM"]).union(test["ITEM"])
    meta_items = set(meta["ITEM"])
    missing = all_items - meta_items
    if missing:
        raise AssertionError(f"Interaction items missing from meta: {len(missing)} e.g. {list(missing)[:10]}")

    # test timestamp never earlier than valid (allow equal)
    merged = valid[["USER", "TIMESTAMP", "ITEM"]].rename(
        columns={"TIMESTAMP": "ts_valid", "ITEM": "item_valid"}
    ).merge(
        test[["USER", "TIMESTAMP", "ITEM"]].rename(
            columns={"TIMESTAMP": "ts_test", "ITEM": "item_test"}
        ),
        on="USER",
    )
    earlier = merged[merged["ts_test"] < merged["ts_valid"]]
    if len(earlier) > 0:
        raise AssertionError(f"test timestamp earlier than valid for {len(earlier)} users")

    return {
        "n_users": len(train_users),
        "n_train_interactions": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
        "n_meta_items": len(meta_items),
        "n_interaction_items": len(all_items),
        "users_equal": True,
        "valid_one_per_user": True,
        "test_one_per_user": True,
        "items_subseteq_meta": True,
        "test_ts_ge_valid": True,
    }


def build_user_sequences(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, Any]:
    """Build per-user sequences sorted by TIMESTAMP; same-stamp keeps file order."""

    def _seq_from_df(df: pd.DataFrame) -> dict[int, list[dict]]:
        # stable sort: TIMESTAMP then original row order
        d = df.copy()
        d["_ord"] = range(len(d))
        d = d.sort_values(["USER", "TIMESTAMP", "_ord"], kind="mergesort")
        out: dict[int, list[dict]] = {}
        for uid, g in d.groupby("USER", sort=False):
            out[int(uid)] = [
                {
                    "item": int(r.ITEM),
                    "rating": float(r.RATING),
                    "timestamp": float(r.TIMESTAMP),
                }
                for r in g.itertuples(index=False)
            ]
        return out

    train_seq = _seq_from_df(train)
    valid_target = {
        int(r.USER): {"item": int(r.ITEM), "rating": float(r.RATING), "timestamp": float(r.TIMESTAMP)}
        for r in valid.itertuples(index=False)
    }
    test_target = {
        int(r.USER): {"item": int(r.ITEM), "rating": float(r.RATING), "timestamp": float(r.TIMESTAMP)}
        for r in test.itertuples(index=False)
    }
    return {
        "train_seq": train_seq,
        "valid_target": valid_target,
        "test_target": test_target,
    }


def compute_stats(train_seq: dict[int, list], n_items: int) -> dict[str, Any]:
    lengths = [len(v) for v in train_seq.values()]
    n_inter = sum(lengths)
    return {
        "n_users": len(train_seq),
        "n_items": n_items,
        "n_train_interactions": n_inter,
        "avg_train_seq_len": float(n_inter / max(len(lengths), 1)),
        "min_train_seq_len": int(min(lengths)) if lengths else 0,
        "max_train_seq_len": int(max(lengths)) if lengths else 0,
        "median_train_seq_len": float(sorted(lengths)[len(lengths) // 2]) if lengths else 0.0,
    }


def load_and_process(
    data_cfg_path: str | Path,
    *,
    write_processed: bool = True,
) -> dict[str, Any]:
    """Full pipeline: validate, build sequences, optionally write processed artifacts."""
    cfg = load_data_config(data_cfg_path)
    raw_dir = Path(cfg["raw_dir"])
    processed_dir = Path(cfg["processed_dir"])
    manifest_path = Path(cfg.get("manifest", raw_dir / "manifest.json"))
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest_report = validate_against_manifest(raw_dir, manifest)

    train = _read_tsv(raw_dir / "train.txt")
    valid = _read_tsv(raw_dir / "valid.txt")
    test = _read_tsv(raw_dir / "test.txt")
    meta = _read_tsv(raw_dir / "item_meta.txt")

    for col in INTERACTION_COLS:
        if col not in train.columns:
            raise ValueError(f"train missing column {col}")
    for col in META_COLS:
        if col not in meta.columns:
            raise ValueError(f"item_meta missing column {col}")

    invariants = _check_split_invariants(train, valid, test, meta)
    seqs = build_user_sequences(train, valid, test)
    n_items = int(meta["ITEM"].max()) + 1  # IDs are 0-contiguous
    # ensure n_items covers all
    n_items = max(n_items, int(meta["ITEM"].nunique()), int(max(meta["ITEM"])) + 1)

    text_fields = tuple(cfg.get("text_fields", list(DEFAULT_TEXT_FIELDS)))
    texts = item_text_table(meta, text_fields)

    stats = compute_stats(seqs["train_seq"], n_items=n_items)
    stats.update(
        {
            "n_valid": invariants["n_valid"],
            "n_test": invariants["n_test"],
            "n_meta_rows": len(meta),
            "text_fields": list(text_fields),
            "holdout_h": int(cfg.get("holdout_h", 3)),
            "max_len": int(cfg.get("max_len", 50)),
            "raw_dir": str(raw_dir),
        }
    )

    bundle = {
        "train_seq": seqs["train_seq"],
        "valid_target": seqs["valid_target"],
        "test_target": seqs["test_target"],
        "item_text": texts,
        "n_users": stats["n_users"],
        "n_items": n_items,
        "stats": stats,
        "manifest_report": manifest_report,
        "invariants": invariants,
        "cfg": cfg,
    }

    if write_processed:
        processed_dir.mkdir(parents=True, exist_ok=True)
        pkl_path = processed_dir / "beauty_loo.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

        # also write light parquet tables for inspection
        train.to_parquet(processed_dir / "train.parquet", index=False)
        valid.to_parquet(processed_dir / "valid.parquet", index=False)
        test.to_parquet(processed_dir / "test.parquet", index=False)
        meta.to_parquet(processed_dir / "item_meta.parquet", index=False)

        stats_path = processed_dir / "stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        # hash of processed pickle
        proc_hash = sha256_file(pkl_path)
        hash_doc = {
            "beauty_loo.pkl_sha256": proc_hash,
            "manifest_validation": {k: v["ok"] for k, v in manifest_report.items()},
            "stats": stats,
        }
        with open(processed_dir / "hash.json", "w", encoding="utf-8") as f:
            json.dump(hash_doc, f, indent=2, ensure_ascii=False)

        bundle["processed_dir"] = str(processed_dir)
        bundle["processed_pkl"] = str(pkl_path)
        bundle["processed_hash"] = proc_hash

    return bundle


def load_processed(processed_dir: str | Path) -> dict[str, Any]:
    pkl_path = Path(processed_dir) / "beauty_loo.pkl"
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def subset_users(bundle: dict[str, Any], max_users: int | None, seed: int = 42) -> dict[str, Any]:
    """Optionally keep a deterministic subset of users for smoke training."""
    if max_users is None or max_users <= 0 or max_users >= bundle["n_users"]:
        return bundle
    import random

    rng = random.Random(seed)
    users = sorted(bundle["train_seq"].keys())
    chosen = set(rng.sample(users, max_users))
    out = dict(bundle)
    out["train_seq"] = {u: bundle["train_seq"][u] for u in chosen}
    out["valid_target"] = {u: bundle["valid_target"][u] for u in chosen}
    out["test_target"] = {u: bundle["test_target"][u] for u in chosen}
    out["n_users"] = len(chosen)
    out["subset"] = {"max_users": max_users, "seed": seed, "n_users": len(chosen)}
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/data/beauty.yaml"))
    args = ap.parse_args()
    bundle = load_and_process(args.config, write_processed=True)
    print(json.dumps(bundle["stats"], indent=2))
    print("manifest_ok:", all(v["ok"] for v in bundle["manifest_report"].values()))
    print("processed:", bundle.get("processed_pkl"))
