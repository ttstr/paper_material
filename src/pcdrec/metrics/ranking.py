"""Full-catalog ranking metrics: HR@K and NDCG@K."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def hit_at_k(rank: int, k: int) -> float:
    """rank is 1-based position of the ground-truth item; 0 or negative means not found."""
    if rank <= 0:
        return 0.0
    return 1.0 if rank <= k else 0.0


def ndcg_at_k(rank: int, k: int) -> float:
    """Binary relevance NDCG@K with 1-based rank."""
    if rank <= 0 or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def ranks_from_scores(scores: np.ndarray, target: int) -> int:
    """Compute 1-based rank of target among scores (higher is better). Ties: target loses (worse rank)."""
    target_score = scores[target]
    if not np.isfinite(target_score):
        return int(scores.shape[0])
    finite = np.isfinite(scores)
    # treat non-finite competitors as not better
    better = int(np.sum((scores > target_score) & finite))
    ties = int(np.sum((scores == target_score) & finite)) - 1
    if ties < 0:
        ties = 0
    return better + ties + 1


def evaluate_ranks(
    ranks: Sequence[int],
    ks: Iterable[int] = (5, 10, 20),
) -> dict[str, float]:
    """Aggregate HR/NDCG from a list of 1-based ranks."""
    ks = list(ks)
    n = max(len(ranks), 1)
    out: dict[str, float] = {}
    for k in ks:
        hrs = [hit_at_k(r, k) for r in ranks]
        nds = [ndcg_at_k(r, k) for r in ranks]
        out[f"hr@{k}"] = float(sum(hrs) / n)
        out[f"ndcg@{k}"] = float(sum(nds) / n)
    out["n"] = float(len(ranks))
    out["mean_rank"] = float(sum(ranks) / n) if ranks else 0.0
    return out


def metrics_from_score_matrix(
    scores: np.ndarray,
    targets: Sequence[int],
    ks: Iterable[int] = (5, 10, 20),
) -> dict[str, float]:
    """scores: [B, n_items]; targets: length B."""
    ranks = [ranks_from_scores(scores[i], int(targets[i])) for i in range(len(targets))]
    return evaluate_ranks(ranks, ks)
