"""Resumable JSONL cache of LLM calls + call/token accounting (ledger)."""

from __future__ import annotations

import json
import time
from pathlib import Path


class CallCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: dict[str, dict] = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        self.records[r["key"]] = r

    def get(self, key: str) -> dict | None:
        return self.records.get(key)

    def put(self, rec: dict) -> None:
        rec = {**rec, "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        self.records[rec["key"]] = rec
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def ledger(self) -> dict:
        recs = list(self.records.values())
        out: dict = {"calls": len(recs)}
        by_stage: dict = {}
        for r in recs:
            s = by_stage.setdefault(r.get("stage", "?"), {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                                           "seconds": 0.0, "parse_ok": 0})
            s["calls"] += 1
            s["prompt_tokens"] += int(r.get("prompt_tokens", 0))
            s["completion_tokens"] += int(r.get("completion_tokens", 0))
            s["seconds"] += float(r.get("seconds", 0.0))
            s["parse_ok"] += int(bool(r.get("parse_ok")))
        for s in by_stage.values():
            s["sec_per_call"] = s["seconds"] / max(s["calls"], 1)
            s["completion_tok_per_sec"] = s["completion_tokens"] / s["seconds"] if s["seconds"] else None
            s["total_tok_per_sec"] = (s["prompt_tokens"] + s["completion_tokens"]) / s["seconds"] if s["seconds"] else None
            s["parse_rate"] = s["parse_ok"] / max(s["calls"], 1)
        out["by_stage"] = by_stage
        out["prompt_tokens"] = sum(s["prompt_tokens"] for s in by_stage.values())
        out["completion_tokens"] = sum(s["completion_tokens"] for s in by_stage.values())
        out["seconds"] = sum(s["seconds"] for s in by_stage.values())
        return out
