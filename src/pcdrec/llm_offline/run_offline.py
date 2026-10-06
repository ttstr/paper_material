"""Run the offline LLM stage: profile extraction then P-permutation teacher ranking (resumable).

  python -m pcdrec.llm_offline.run_offline --config configs/llm/qwen2.5-1.5b-cpu.yaml --n-users 60 \
      --out results/pilot/llm   [--stage profile|rank|all]

Outputs (under --out): inputs.jsonl (per-user leakage-checked inputs), calls.jsonl (raw call cache:
prompt hash, tokens, seconds, raw text), profiles.jsonl, rankings.jsonl, ledger.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from pcdrec.data.load_splits import load_data_config, load_processed
from pcdrec.llm_offline.backends import CachedOnlyBackend, make_backend
from pcdrec.llm_offline.build_inputs import check_no_leakage, permutations, select_users, split_user, stage0_candidates
from pcdrec.llm_offline.cache import CallCache
from pcdrec.llm_offline.prompts import messages_hash, profile_messages, rank_messages
from pcdrec.llm_offline.schema import PROFILE_SCHEMA, RANK_SCHEMA, extract_json, parse_profile_text, parse_ranking_text
from pcdrec.paths import REPO_ROOT, resolve


def load_stage0(ckpt_path: Path, n_items: int, data_cfg: dict):
    from pcdrec.train import build_model

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])
    cfg.setdefault("model", "sasrec")
    m = build_model(cfg, n_items, 1, int(ck["max_len"]), data_cfg)
    m.load_state_dict(ck["model_state"])
    return m.eval(), int(ck["max_len"])


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_calls(backend, cache: CallCache, jobs: list[dict], batch_size: int, log_every: int = 1) -> None:
    """jobs: dicts with key, messages, max_tokens, schema, meta. Results stored in cache."""
    todo = [j for j in jobs if cache.get(j["key"]) is None]
    print(f"[llm] {len(jobs) - len(todo)} cached, {len(todo)} to run", flush=True)
    if getattr(backend, "cached_only", False):
        if todo:
            print(f"[llm] cached-only: skipping {len(todo)} uncached calls", flush=True)
        return
    t0 = time.time()
    for s in range(0, len(todo), batch_size):
        chunk = todo[s : s + batch_size]
        gens = backend.generate_batch([j["messages"] for j in chunk], max_tokens=max(j["max_tokens"] for j in chunk),
                                      json_schema=chunk[0]["schema"])
        for j, g in zip(chunk, gens):
            parsed = extract_json(g.text)
            cache.put({"key": j["key"], **j["meta"], "model": backend.model_name, "decoding": backend.decoding(),
                       "prompt_tokens": g.prompt_tokens, "completion_tokens": g.completion_tokens,
                       "seconds": g.seconds, "batch_size": len(chunk), "text": g.text,
                       "parse_ok": parsed is not None})
        done = s + len(chunk)
        if done % log_every == 0 or done == len(todo):
            el = time.time() - t0
            print(f"[llm] {done}/{len(todo)} calls, {el:.0f}s elapsed, {el / done:.1f}s/call", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-config", default=str(REPO_ROOT / "configs/data/beauty.yaml"))
    ap.add_argument("--n-users", type=int, default=None)
    ap.add_argument("--users-file", default=None, help="optional json list of user ids")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=["profile", "rank", "all"], default="all")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--cached-only", action="store_true",
                    help="no model is loaded; rebuild profiles/rankings from calls.jsonl, skipping uncached calls")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    llm_cfg, off = cfg["llm"], cfg["offline"]
    if args.threads:
        llm_cfg["threads"] = args.threads
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data_cfg = load_data_config(args.data_config)
    bundle = load_processed(data_cfg["processed_dir"])

    if args.users_file:
        users = json.load(open(args.users_file))
    else:
        users = select_users(bundle, args.n_users or int(off.get("n_users", 50)), seed=int(off.get("user_seed", 42)),
                             min_train_len=int(off.get("min_train_len", 6)))
    rows = [split_user(bundle, u, h=int(off["holdout_h"]), n_hist=int(off["n_history"])) for u in users]
    stage0, max_len = load_stage0(resolve(off["stage0_checkpoint"]), int(bundle["n_items"]), data_cfg)
    stage0_candidates(stage0, rows, M=int(off["cand_M"]), max_len=max_len)
    for r in rows:
        check_no_leakage(r)  # hard stop before any LLM call
        r["perms"] = permutations(r["candidates"], int(off["n_perm"]), int(off.get("perm_seed", 0)), r["user"])
    write_jsonl(out / "inputs.jsonl", rows)

    backend = CachedOnlyBackend(llm_cfg) if args.cached_only else make_backend(llm_cfg)
    cache = CallCache(out / "calls.jsonl")
    bs = int(llm_cfg.get("batch_size", 1))
    dec = backend.decoding()
    texts = bundle["item_text"]

    # ---- stage 1: profiles
    pjobs = []
    for r in rows:
        msgs = profile_messages(r["history"], texts, max_claims=int(off["max_claims"]))
        key = messages_hash(msgs, backend.model_name, {**dec, "stage": "profile"})
        pjobs.append({"key": key, "messages": msgs, "max_tokens": int(off["profile_max_tokens"]),
                      "schema": PROFILE_SCHEMA, "meta": {"stage": "profile", "user": r["user"], "prompt_hash": key}})
    if args.stage in ("profile", "all"):
        run_calls(backend, cache, pjobs, bs)
    profiles = []
    for r, j in zip(rows, pjobs):
        rec = cache.get(j["key"])
        if rec is None:
            continue
        claims, rep = parse_profile_text(rec["text"], int(off["max_claims"]))
        profiles.append({"user": r["user"], "claims": claims, "report": rep, "prompt_hash": j["key"]})
    write_jsonl(out / "profiles.jsonl", profiles)
    if args.stage == "profile":
        (out / "ledger.json").write_text(json.dumps(cache.ledger(), indent=2))
        return
    prof_by_u = {p["user"]: p for p in profiles}

    # ---- stage 2: teacher ranking under P permutations (reasons only for perm 0, top-k)
    rjobs = []
    for r in rows:
        if r["user"] not in prof_by_u:
            continue
        claims = prof_by_u[r["user"]]["claims"]
        for p, perm in enumerate(r["perms"]):
            k = int(off["reasons_top_k"]) if p in off.get("reasons_in_perms", [0]) else 0
            msgs = rank_messages(claims, perm, texts, reasons_top_k=k)
            key = messages_hash(msgs, backend.model_name, {**dec, "stage": "rank"})
            rjobs.append({"key": key, "messages": msgs,
                          "max_tokens": int(off["rank_max_tokens"]) + (int(off["reason_max_tokens"]) if k else 0),
                          "schema": RANK_SCHEMA,
                          "meta": {"stage": "rank", "user": r["user"], "perm": p, "prompt_hash": key}})
    # user-major order: an interrupted run still leaves complete users
    rjobs.sort(key=lambda j: (j["meta"]["user"], j["meta"]["perm"]))
    run_calls(backend, cache, rjobs, bs)
    rankings = []
    for j in rjobs:
        rec = cache.get(j["key"])
        if rec is None:
            continue
        r = next(x for x in rows if x["user"] == j["meta"]["user"])
        perm = r["perms"][j["meta"]["perm"]]
        ranking, reasons, rep = parse_ranking_text(rec["text"], perm)
        rankings.append({"user": r["user"], "perm": j["meta"]["perm"], "presented": perm, "ranking": ranking,
                         "reasons": reasons, "report": rep, "prompt_hash": j["key"]})
    # keep only users with all P permutations (an interrupted / cached-only run may leave one partial user)
    n_by_u: dict = {}
    for x in rankings:
        n_by_u[x["user"]] = n_by_u.get(x["user"], 0) + 1
    rankings = [x for x in rankings if n_by_u[x["user"]] == int(off["n_perm"])]
    rankings.sort(key=lambda x: (x["user"], x["perm"]))
    write_jsonl(out / "rankings.jsonl", rankings)
    led = cache.ledger()
    led.update({"model": backend.model_name, "backend": backend.name, "dtype": llm_cfg.get("dtype"),
                "threads": llm_cfg.get("threads"), "batch_size": bs, "n_users": len(rows)})
    (out / "ledger.json").write_text(json.dumps(led, indent=2))
    print(json.dumps(led, indent=2))


if __name__ == "__main__":
    main()
