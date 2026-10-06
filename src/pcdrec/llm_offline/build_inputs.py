"""Construct leakage-safe LLM inputs per user (plan §2.2).

For user u with train sequence S = (i_1..i_n) (valid/test held out separately):
  profile history  = S[:-h][-n_hist:]          (h probe items are never shown to the profile LLM)
  probes           = S[-h:]                    (used only by the T score)
  teacher target   = S[-1]                     (= last probe; the student is distilled at this position)
  candidates C_u   = Stage-0 SASRec top-M given S[:-1], minus the user's valid/test targets and
                     minus the other probes S[-h:-1]; the teacher target is inserted if missing.
Every prompt is checked with ``assert_no_target_ids_in_llm_input`` before any LLM call.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from pcdrec.data.leakage_check import assert_no_target_ids_in_llm_input


def select_users(bundle: dict, n_users: int, seed: int = 42, min_train_len: int = 6) -> list[int]:
    elig = sorted(u for u, s in bundle["train_seq"].items() if len(s) >= min_train_len)
    rng = random.Random(seed)
    return sorted(rng.sample(elig, min(n_users, len(elig))))


def split_user(bundle: dict, u: int, h: int = 3, n_hist: int = 20) -> dict:
    items = [e["item"] for e in bundle["train_seq"][u]]
    assert len(items) > h, f"user {u} too short for holdout h={h}"
    prefix = items[:-h]
    return {
        "user": int(u),
        "history": prefix[-n_hist:],
        "prefix_full": prefix,
        "probes": items[-h:],
        "teacher_target": items[-1],
        "teacher_context": items[:-1],
        "forbidden": sorted({int(bundle["valid_target"][u]["item"]), int(bundle["test_target"][u]["item"])}),
    }


@torch.no_grad()
def stage0_candidates(model, rows: list[dict], M: int = 20, max_len: int = 50, batch_size: int = 64) -> None:
    """Fill rows[*]['candidates'] (sorted by Stage-0 score) in place."""
    from pcdrec.evaluator import pad_history

    model.eval()
    for s in range(0, len(rows), batch_size):
        chunk = rows[s : s + batch_size]
        seqs = torch.stack([pad_history(r["teacher_context"], max_len, model.pad_id) for r in chunk])
        scores = model.predict_logits(seqs)
        top = torch.topk(scores, k=M + 2 * len(chunk[0]["probes"]) + 4, dim=1).indices.tolist()
        for r, cand in zip(chunk, top):
            banned = set(r["forbidden"]) | set(r["probes"][:-1])
            c = [int(x) for x in cand if int(x) not in banned and int(x) != r["teacher_target"]][: M - 1]
            gt_rank = None
            full = [int(x) for x in cand if int(x) not in banned]
            if r["teacher_target"] in full[:M]:
                gt_rank = full.index(r["teacher_target"])
            # insert GT at its Stage-0 position if it was in the top-M, else at the end (order is shuffled later)
            pos = gt_rank if gt_rank is not None else len(c)
            c.insert(min(pos, len(c)), int(r["teacher_target"]))
            r["candidates"] = c[:M]
            r["gt_in_stage0_topM"] = gt_rank is not None


def check_no_leakage(row: dict) -> None:
    assert_no_target_ids_in_llm_input(row["history"], row["forbidden"], context=f"profile u={row['user']}")
    assert_no_target_ids_in_llm_input(row["history"], row["probes"], context=f"profile-probes u={row['user']}")
    if "candidates" in row:
        assert_no_target_ids_in_llm_input(row["candidates"], row["forbidden"], context=f"rank u={row['user']}")
        other_probes = set(row["probes"][:-1]) - {row["teacher_target"]}
        assert_no_target_ids_in_llm_input(row["candidates"], other_probes, context=f"rank-probes u={row['user']}")


def permutations(cands: list[int], P: int, seed: int, user: int) -> list[list[int]]:
    rng = np.random.default_rng([seed, user])
    return [list(np.asarray(cands)[rng.permutation(len(cands))].tolist()) for _ in range(P)]
