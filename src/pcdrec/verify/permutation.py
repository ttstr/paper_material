"""R: permutation stability = mean pairwise Kendall-tau between the P teacher rankings, mapped to [0,1]."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.stats import kendalltau


def kendall(rank_a: list[int], rank_b: list[int]) -> float:
    pos_b = {it: i for i, it in enumerate(rank_b)}
    common = [it for it in rank_a if it in pos_b]
    if len(common) < 2:
        return 0.0
    tau = kendalltau(np.arange(len(common)), [pos_b[it] for it in common]).statistic
    return 0.0 if np.isnan(tau) else float(tau)


def permutation_stability(rankings: list[list[int]]) -> dict:
    if len(rankings) < 2:
        return {"R": 1.0, "tau_mean": 1.0, "n_pairs": 0}
    taus = [kendall(a, b) for a, b in combinations(rankings, 2)]
    t = float(np.mean(taus))
    return {"R": (t + 1.0) / 2.0, "tau_mean": t, "n_pairs": len(taus)}


def position_bias(presented: list[list[int]], rankings: list[list[int]]) -> float:
    """Kendall-tau between presentation order and teacher order, averaged (0 = no position bias)."""
    return float(np.mean([kendall(p, r) for p, r in zip(presented, rankings)])) if rankings else 0.0
