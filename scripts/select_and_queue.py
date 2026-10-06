#!/usr/bin/env python
"""Pick each method's configuration by seed-42 valid NDCG@10 (configs/search/baselines.yaml),
write results/search_summary.csv, and (optionally) append the remaining seeds to a run queue.

  python scripts/select_and_queue.py                      # summary only
  python scripts/select_and_queue.py --queue logs/queue.txt --methods BERT4Rec GRU4Rec
A method is selected only when all its grid runs exist; nothing is queued twice.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load(tag: str, seed: int) -> dict | None:
    p = ROOT / "results" / f"{tag}_s{seed}_metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def selection(cfg: dict) -> dict:
    out = {}
    s0 = int(cfg["search_seed"])
    for name, m in cfg["methods"].items():
        rows = []
        for g in m["grid"]:
            d = load(g["tag"], s0)
            rows.append({**g, "valid_ndcg@10": d["valid"]["ndcg@10"] if d else None,
                         "best_epoch": d["best_epoch"] if d else None, "elapsed_sec": d["elapsed_sec"] if d else None})
        complete = all(r["valid_ndcg@10"] is not None for r in rows)
        best = max(rows, key=lambda r: r["valid_ndcg@10"]) if complete else None
        out[name] = {"model": m["model"], "grid": rows, "complete": complete, "selected": best}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/search/baselines.yaml"))
    ap.add_argument("--queue", default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    sel = selection(cfg)
    with open(ROOT / "results/search_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "config", "tag", "seed", "valid_ndcg@10", "best_epoch", "elapsed_sec", "selected"])
        for name, s in sel.items():
            for r in s["grid"]:
                w.writerow([name, r["label"], r["tag"], cfg["search_seed"], r["valid_ndcg@10"], r["best_epoch"],
                            r["elapsed_sec"], bool(s["selected"] and s["selected"]["tag"] == r["tag"])])
    for name, s in sel.items():
        print(name, "complete" if s["complete"] else "incomplete", "->", s["selected"]["label"] if s["selected"] else "-",
              [(r["label"], r["valid_ndcg@10"]) for r in s["grid"]])
    if args.queue:
        import fcntl

        q = Path(args.queue)
        hist_path = q.with_name(q.stem + "_history.txt")  # every tag ever queued (a popped, running job is not in q)
        with open(str(q) + ".lock", "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)  # same lock the workers use when popping
            existing = q.read_text() if q.exists() else ""
            history = set(hist_path.read_text().split()) if hist_path.exists() else set()
            lines = []
            for name, s in sel.items():
                if args.methods and name not in args.methods or not s["selected"]:
                    continue
                for seed in cfg["seeds"]:
                    tag = f"{s['selected']['tag']}_s{seed}"
                    if (load(s["selected"]["tag"], seed) or f"{tag} " in existing or tag in history
                            or (ROOT / "logs" / f"{tag}.log").exists()):
                        continue
                    lines.append(f"{tag} {s['model']} {seed} {s['selected']['args']}".rstrip())
            with open(q, "a") as f:
                for l in lines:
                    f.write(l + "\n")
            with open(hist_path, "a") as f:
                for l in lines:
                    f.write(l.split()[0] + "\n")
        print(f"queued {len(lines)} runs:", *lines, sep="\n  ")


if __name__ == "__main__":
    main()
