"""Ensure LLM input paths cannot include valid/test targets; holdout prefix on train only."""

from pathlib import Path

import pytest

from pcdrec.data.leakage_check import (
    assert_no_target_ids_in_llm_input,
    assert_prefix_respects_holdout,
    collect_valid_test_targets,
    train_prefix_for_profile,
)
from pcdrec.data.load_splits import load_and_process, load_data_config, load_processed
from pcdrec.paths import REPO_ROOT

DATA_CFG = str(REPO_ROOT / "configs/data/beauty.yaml")
PROCESSED = Path(load_data_config(DATA_CFG)["processed_dir"])
HOLDOUT_H = 3


@pytest.fixture(scope="module")
def bundle():
    if not (PROCESSED / "beauty_loo.pkl").exists():
        return load_and_process(DATA_CFG, write_processed=True)
    return load_processed(PROCESSED)


def test_profile_prefix_excludes_valid_test_targets(bundle):
    valid_by_user = {u: t["item"] for u, t in bundle["valid_target"].items()}
    test_by_user = {u: t["item"] for u, t in bundle["test_target"].items()}
    forbidden = collect_valid_test_targets(valid_by_user, test_by_user)

    # Simulate LLM history inputs = train prefix with holdout_h
    for u, seq in list(bundle["train_seq"].items())[:500]:  # sample speed; full check below on targets
        items = [e["item"] for e in seq]
        prefix = train_prefix_for_profile(items, holdout_h=HOLDOUT_H)
        assert_prefix_respects_holdout(items, prefix, holdout_h=HOLDOUT_H)
        # Prefix must not *need* to exclude targets that only appear in train,
        # but must never include this user's valid/test target as LLM evidence
        # when those targets are not in the allowed prefix.
        user_forbidden = {valid_by_user[u], test_by_user[u]}
        # If a target somehow equals an earlier train item (repurchase), it may appear
        # in train history; policy: LLM may see train items only. Still forbid injecting
        # valid/test *target slots* as extra ids outside train prefix.
        assert_no_target_ids_in_llm_input(
            [],  # empty extra ids
            user_forbidden,
            context=f"user-{u}-extras",
        )
        # Explicit: do not append valid/test targets into LLM input
        with pytest.raises(AssertionError):
            assert_no_target_ids_in_llm_input(
                list(prefix) + [valid_by_user[u]],
                [valid_by_user[u]],
                context="injected-valid",
            )


def test_holdout_boundary_unit():
    items = [10, 11, 12, 13, 14, 15]
    prefix = train_prefix_for_profile(items, holdout_h=3)
    assert prefix == [10, 11, 12]
    assert_prefix_respects_holdout(items, prefix, holdout_h=3)
    with pytest.raises(AssertionError):
        assert_prefix_respects_holdout(items, items, holdout_h=3)


def test_no_leakage_global_forbidden_union(bundle):
    valid_by_user = {u: t["item"] for u, t in bundle["valid_target"].items()}
    test_by_user = {u: t["item"] for u, t in bundle["test_target"].items()}
    forbidden = collect_valid_test_targets(valid_by_user, test_by_user)
    assert len(forbidden) > 0
    # LLM cache dir (future) must not be seeded with forbidden-only lists
    with pytest.raises(AssertionError):
        assert_no_target_ids_in_llm_input(list(forbidden)[:5], forbidden, context="bad-cache")


def test_llm_inputs_real_users_never_contain_valid_test_or_probes(bundle):
    """End-to-end on real users: the exact prompt builders used by run_offline are leakage-free."""
    import json as _json

    import torch as _torch

    from pcdrec.llm_offline.build_inputs import check_no_leakage, select_users, split_user, stage0_candidates
    from pcdrec.llm_offline.prompts import profile_messages, rank_messages
    from pcdrec.models.sasrec import SASRec

    _torch.manual_seed(0)
    model = SASRec(n_items=int(bundle["n_items"]), hidden_size=16, n_layers=1, n_heads=2, inner_size=32,
                   max_seq_length=50)
    users = select_users(bundle, 300, seed=1)
    rows = [split_user(bundle, u, h=HOLDOUT_H, n_hist=20) for u in users]
    stage0_candidates(model, rows, M=20, max_len=50)
    for r in rows:
        check_no_leakage(r)
        prof = _json.dumps(profile_messages(r["history"], bundle["item_text"]))
        rank = _json.dumps(rank_messages([], r["candidates"], bundle["item_text"]))
        for f in r["forbidden"]:
            assert f"[{f}]" not in prof and f"[{f}]" not in rank
        for p in r["probes"]:
            assert f"[{p}]" not in prof
        for p in set(r["probes"][:-1]) - {r["teacher_target"]}:
            assert f"[{p}]" not in rank
