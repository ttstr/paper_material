"""The single shared evaluator for every method (full-catalogue ranking, history NOT filtered).

Protocol (unchanged from Stage-0):
- score all n_items catalogue items (no candidate sampling, no history filtering);
- valid: input = train history; test: input = train history + valid item;
- rank = 1 + #items with a strictly higher score + #other items with an equal score
  (ties count against the target), HR@K / NDCG@K with binary relevance.

Every model exposes ``score(users, item_seq) -> [B, n_items]`` (sequential models
ignore ``users``; MF-BPR ignores ``item_seq``).
"""

from __future__ import annotations

import numpy as np
import torch

from pcdrec.metrics.ranking import evaluate_ranks, ranks_from_score_matrix


def pad_history(items: list[int], max_len: int, pad_id: int) -> torch.Tensor:
    if len(items) > max_len:
        items = items[-max_len:]
    padded = [pad_id] * (max_len - len(items)) + items
    return torch.tensor(padded, dtype=torch.long)


def history_for(u: int, train_seq: dict, history_mode: str, valid_target: dict | None) -> list[int]:
    hist = [e["item"] for e in train_seq[u]]
    if history_mode == "train_valid":
        assert valid_target is not None
        hist = hist + [valid_target[u]["item"]]
    return hist


@torch.no_grad()
def evaluate_split(
    model,
    train_seq: dict,
    targets: dict,
    max_len: int,
    device: torch.device,
    ks=(5, 10, 20),
    batch_size: int = 128,
    history_mode: str = "train",  # train | train_valid
    valid_target: dict | None = None,
    return_ranks: bool = False,
):
    model.eval()
    users = sorted(targets.keys())
    pad_id = model.pad_id
    all_ranks: list[np.ndarray] = []
    for start in range(0, len(users), batch_size):
        bu = users[start : start + batch_size]
        seqs = torch.stack([pad_history(history_for(u, train_seq, history_mode, valid_target), max_len, pad_id)
                            for u in bu]).to(device)
        u_t = torch.tensor(bu, dtype=torch.long, device=device)
        tg = torch.tensor([int(targets[u]["item"]) for u in bu], dtype=torch.long, device=device)
        scores = model.score(u_t, seqs).float()
        all_ranks.append(ranks_from_score_matrix(scores, tg).cpu().numpy())
    ranks = np.concatenate(all_ranks) if all_ranks else np.zeros(0, dtype=np.int64)
    metrics = evaluate_ranks([int(r) for r in ranks], ks)
    if return_ranks:
        return metrics, np.asarray(users, dtype=np.int64), ranks.astype(np.int64)
    return metrics
