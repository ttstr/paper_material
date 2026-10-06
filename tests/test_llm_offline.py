"""Offline LLM stage: schema parsing, resumable cache/ledger, backend config rules, leakage-safe inputs."""

import json
from pathlib import Path

import pytest
import torch

from pcdrec.data.leakage_check import assert_no_target_ids_in_llm_input
from pcdrec.llm_offline.backends import OpenAICompatBackend, make_backend
from pcdrec.llm_offline.build_inputs import check_no_leakage, permutations, split_user, stage0_candidates
from pcdrec.llm_offline.cache import CallCache
from pcdrec.llm_offline.prompts import messages_hash, profile_messages, rank_messages
from pcdrec.llm_offline.schema import parse_profile_text, parse_ranking_text, validate_ranking
from pcdrec.models.sasrec import SASRec


def test_parse_profile_fenced_and_truncated():
    txt = '```json\n{"claims": [{"aspect": "brand", "statement": "likes NYX", "polarity": "+", "evidence": [3, "7"]}]}\n```'
    claims, rep = parse_profile_text(txt)
    assert rep["parsed"] and not rep["salvaged"] and claims[0]["evidence"] == [3, 7]
    trunc = '{"claims": [{"aspect": "a", "statement": "s1", "polarity": "-", "evidence": [1]}, {"aspect": "b", "statem'
    claims, rep = parse_profile_text(trunc)
    assert rep["salvaged"] and len(claims) == 1 and claims[0]["polarity"] == "-"
    claims, rep = parse_profile_text("I cannot help")
    assert claims == [] and not rep["parsed"]


def test_parse_ranking_dedup_unknown_missing():
    cands = [10, 11, 12, 13]
    rk, reasons, rep = parse_ranking_text('{"ranking": [12, 12, 99, 10], "reasons": [{"item": 12, "claims": [0], "reason": "x"}]}', cands)
    assert rk == [12, 10, 11, 13] and rep["n_dup"] == 1 and rep["n_unknown"] == 1 and rep["n_missing_appended"] == 2
    assert reasons[0]["item"] == 12
    rk, _, rep = parse_ranking_text('{"ranking": [13, 11, 13, 13, 13, 1', cands)
    assert rep["salvaged"] and rk[:2] == [13, 11] and sorted(rk) == cands


def test_cache_resume_and_ledger(tmp_path):
    c = CallCache(tmp_path / "calls.jsonl")
    c.put({"key": "a", "stage": "profile", "prompt_tokens": 100, "completion_tokens": 50, "seconds": 2.0, "parse_ok": True})
    c.put({"key": "b", "stage": "rank", "prompt_tokens": 10, "completion_tokens": 5, "seconds": 1.0, "parse_ok": False})
    c2 = CallCache(tmp_path / "calls.jsonl")
    assert c2.get("a") is not None and c2.get("zzz") is None
    led = c2.ledger()
    assert led["calls"] == 2 and led["by_stage"]["profile"]["completion_tok_per_sec"] == 25.0
    assert led["by_stage"]["rank"]["parse_rate"] == 0.0


def test_api_key_only_from_env(monkeypatch):
    monkeypatch.delenv("PCDREC_TEST_KEY", raising=False)
    with pytest.raises(RuntimeError):
        OpenAICompatBackend({"name": "m", "api_key_env": "PCDREC_TEST_KEY"})
    monkeypatch.setenv("PCDREC_TEST_KEY", "dummy")
    b = make_backend({"backend": "openai", "name": "m", "api_key_env": "PCDREC_TEST_KEY"})
    assert b.api_key == "dummy"


def test_prompt_hash_stable():
    m = profile_messages([1, 2], {1: "a", 2: "b"})
    assert messages_hash(m, "x", {"t": 0}) == messages_hash(m, "x", {"t": 0})
    assert messages_hash(m, "x", {"t": 0}) != messages_hash(m, "y", {"t": 0})


def _toy_bundle():
    train = {0: [1, 2, 3, 4, 5, 6, 7], 1: [8, 9, 10, 11, 12, 13]}
    return {
        "train_seq": {u: [{"item": i, "timestamp": float(k)} for k, i in enumerate(s)] for u, s in train.items()},
        "valid_target": {0: {"item": 20}, 1: {"item": 21}},
        "test_target": {0: {"item": 22}, 1: {"item": 23}},
        "item_text": {i: f"item {i}" for i in range(30)},
        "n_items": 30,
    }


def test_split_and_candidates_exclude_targets_and_probes():
    b = _toy_bundle()
    torch.manual_seed(0)
    model = SASRec(n_items=30, hidden_size=8, n_layers=1, n_heads=2, inner_size=16, max_seq_length=10)
    with torch.no_grad():  # make the forbidden items the most attractive candidates
        model.item_emb.weight[20:24] += 5.0 * model.item_emb.weight[1]
    rows = [split_user(b, u, h=3, n_hist=20) for u in (0, 1)]
    stage0_candidates(model, rows, M=10, max_len=10)
    for r in rows:
        check_no_leakage(r)
        assert r["teacher_target"] in r["candidates"] and len(r["candidates"]) == 10
        assert not set(r["probes"]) & set(r["history"])
        assert not set(r["forbidden"]) & set(r["candidates"])
        perms = permutations(r["candidates"], 3, 0, r["user"])
        assert all(sorted(p) == sorted(r["candidates"]) for p in perms)
        text = json.dumps(rank_messages([], perms[0], b["item_text"]) + profile_messages(r["history"], b["item_text"]))
        for f in r["forbidden"]:
            assert f"[{f}]" not in text
    bad = dict(rows[0], candidates=rows[0]["candidates"] + [20])
    with pytest.raises(AssertionError):
        check_no_leakage(bad)


def test_cached_only_backend_replays_without_generating(tmp_path):
    from pcdrec.llm_offline.backends import CachedOnlyBackend
    from pcdrec.llm_offline.cache import CallCache
    from pcdrec.llm_offline.run_offline import run_calls

    be = CachedOnlyBackend({"name": "Qwen/Qwen2.5-1.5B-Instruct", "backend": "transformers", "max_tokens": 400})
    assert be.decoding() == {"temperature": 0.0, "max_tokens": 400, "backend": "transformers"}
    cache = CallCache(tmp_path / "calls.jsonl")
    cache.put({"key": "a", "stage": "profile", "text": "{}"})
    jobs = [{"key": "a", "messages": [], "max_tokens": 8, "schema": None, "meta": {"stage": "profile"}},
            {"key": "b", "messages": [], "max_tokens": 8, "schema": None, "meta": {"stage": "profile"}}]
    run_calls(be, cache, jobs, batch_size=2)  # must not raise: uncached "b" is skipped
    assert cache.get("b") is None
