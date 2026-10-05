"""Hand-crafted ranking metric checks."""

import math

import numpy as np

from pcdrec.metrics.ranking import hit_at_k, ndcg_at_k, ranks_from_scores, evaluate_ranks


def test_hit_and_ndcg_basic():
    assert hit_at_k(1, 10) == 1.0
    assert hit_at_k(10, 10) == 1.0
    assert hit_at_k(11, 10) == 0.0
    assert hit_at_k(0, 10) == 0.0
    assert abs(ndcg_at_k(1, 10) - 1.0) < 1e-9
    assert abs(ndcg_at_k(2, 10) - 1.0 / math.log2(3)) < 1e-9
    assert ndcg_at_k(11, 10) == 0.0


def test_ranks_from_scores():
    scores = np.array([0.1, 0.9, 0.5, 0.2], dtype=np.float64)
    # target=1 has highest score -> rank 1
    assert ranks_from_scores(scores, 1) == 1
    # target=2 is second -> rank 2
    assert ranks_from_scores(scores, 2) == 2
    # target=0 is worst -> rank 4
    assert ranks_from_scores(scores, 0) == 4


def test_evaluate_ranks_aggregate():
    # ranks: 1, 2, 11
    m = evaluate_ranks([1, 2, 11], ks=(5, 10))
    assert abs(m["hr@5"] - (2 / 3)) < 1e-9
    assert abs(m["hr@10"] - (2 / 3)) < 1e-9
    expected_ndcg5 = (1.0 + 1.0 / math.log2(3) + 0.0) / 3
    assert abs(m["ndcg@5"] - expected_ndcg5) < 1e-9
