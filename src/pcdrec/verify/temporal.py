"""T: temporal predictive consistency. AUC of sup(P_u, i) for the h held-out probe items vs N_neg
popularity-bucket-matched negatives. sup(P_u, i) = max_k polarity_k * cos(phi(statement_k), phi(x_i))."""

from __future__ import annotations

import numpy as np


def popularity_buckets(train_seq: dict, n_items: int, n_buckets: int = 10) -> np.ndarray:
    pop = np.zeros(n_items, dtype=np.int64)
    for evs in train_seq.values():
        for e in evs:
            pop[e["item"]] += 1
    ranks = pop.argsort(kind="stable").argsort(kind="stable")
    return (ranks * n_buckets // n_items).astype(np.int64)


def support(claim_emb: np.ndarray, polarity: np.ndarray, item_emb: np.ndarray) -> np.ndarray:
    """claim_emb [K,D] (normalised), polarity [K] in {+1,-1}, item_emb [N,D] -> sup [N]."""
    if claim_emb.shape[0] == 0:
        return np.zeros(item_emb.shape[0])
    sims = item_emb @ claim_emb.T  # [N, K]
    return (sims * polarity[None, :]).max(axis=1)


def sample_bucket_negatives(probes: list[int], exclude: set[int], buckets: np.ndarray, n_neg: int,
                            rng: np.random.Generator) -> list[int]:
    by_bucket: dict[int, np.ndarray] = {}
    negs: list[int] = []
    per = max(1, n_neg // max(len(probes), 1))
    for p in probes:
        b = int(buckets[p])
        if b not in by_bucket:
            by_bucket[b] = np.flatnonzero(buckets == b)
        pool = by_bucket[b]
        got = 0
        while got < per:
            c = int(pool[rng.integers(len(pool))])
            if c not in exclude:
                negs.append(c)
                got += 1
    return negs


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def temporal_consistency(claim_emb, polarity, probes, negatives, item_emb) -> float:
    s_pos = support(claim_emb, polarity, item_emb[probes])
    s_neg = support(claim_emb, polarity, item_emb[negatives])
    return auc(s_pos, s_neg)
