"""Teacher distribution q_u: Borda aggregation of the P rankings, q_u(j) ∝ exp(-rank_u(j) / tau_T)."""

from __future__ import annotations

import numpy as np


def borda(rankings: list[list[int]], candidates: list[int]) -> list[int]:
    score = {c: 0.0 for c in candidates}
    M = len(candidates)
    for r in rankings:
        for pos, it in enumerate(r):
            if it in score:
                score[it] += M - pos
    return sorted(candidates, key=lambda c: (-score[c], candidates.index(c)))


def teacher_distribution(rankings: list[list[int]], candidates: list[int], tau_T: float = 2.0) -> dict[int, float]:
    order = borda(rankings, candidates)
    logits = -np.arange(len(order), dtype=np.float64) / tau_T
    p = np.exp(logits - logits.max())
    p /= p.sum()
    return {int(c): float(v) for c, v in zip(order, p)}
