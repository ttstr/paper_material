"""PCS components (G, T, R, F), weights, negatives N1-N3 and teacher distribution."""

import numpy as np

from pcdrec.verify.faithfulness import rationale_faithfulness
from pcdrec.verify.grounding import claim_grounding, user_grounding
from pcdrec.verify.negatives import n1_cross_user, n2_polarity_flip, n3_broken_evidence
from pcdrec.verify.nli import LexicalEntail
from pcdrec.verify.pcs import pcs_score, user_weight
from pcdrec.verify.permutation import permutation_stability
from pcdrec.verify.teacher import borda, teacher_distribution
from pcdrec.verify.temporal import auc, popularity_buckets, sample_bucket_negatives, support

TEXT = {1: "nyx eyeliner black", 2: "nyx lipstick red", 3: "dove shampoo", 9: "probe item"}


def test_grounding_valid_invalid_and_hallucinated():
    claims = [{"statement": "nyx black eyeliner", "polarity": "+", "evidence": [1]},
              {"statement": "likes shampoo", "polarity": "+", "evidence": [9]},   # probe -> not allowed
              {"statement": "no evidence", "polarity": "+", "evidence": []}]
    recs = claim_grounding(claims, [1, 2, 3], TEXT, LexicalEntail())
    assert recs[0]["G"] > 0.5 and recs[1]["G"] == 0 and recs[1]["hallucinated_evidence"] and recs[2]["no_evidence"]
    g = user_grounding(recs)
    assert abs(g["halluc_rate"] - 1 / 3) < 1e-9 and 0 < g["G"] < 1


def test_temporal_auc_and_support():
    item = np.eye(4, dtype=np.float32)
    claim = item[[0]]
    s = support(claim, np.array([1.0]), item)
    assert s[0] == 1 and s[1] == 0
    assert auc(np.array([1.0]), np.array([0.0, 0.0])) == 1.0
    assert auc(np.array([0.0]), np.array([0.0])) == 0.5
    train = {0: [{"item": 0}, {"item": 0}, {"item": 1}], 1: [{"item": 0}]}
    b = popularity_buckets(train, 6, n_buckets=3)
    negs = sample_bucket_negatives([5], {5}, b, 4, np.random.default_rng(0))
    assert all(b[n] == b[5] and n != 5 for n in negs)


def test_permutation_stability_bounds():
    assert permutation_stability([[1, 2, 3], [1, 2, 3], [1, 2, 3]])["R"] == 1.0
    assert permutation_stability([[1, 2, 3], [3, 2, 1]])["R"] == 0.0


def test_faithfulness_requires_valid_citation():
    reasons = [{"item": 1, "claims": [0], "reason": "nyx eyeliner"}, {"item": 2, "claims": [5], "reason": "nyx lipstick"}]
    f = rationale_faithfulness(reasons, 1, TEXT, LexicalEntail())
    assert f["per_item"][0]["F"] == 1.0 and f["per_item"][1]["F"] == 0.0 and f["cite_valid_rate"] == 0.5


def test_pcs_weights_switches():
    cfg = {"enabled": True, "theta": 0.25, "T_w": 0.1, "g_min": 0.1}
    p, w = user_weight(0.8, 0.9, 0.7, cfg)
    assert abs(p - 0.8 * 0.9 * 0.7) < 1e-9 and w > 0.9
    assert user_weight(0.05, 0.9, 0.9, cfg)[1] == 0.0          # hard filter
    assert user_weight(0.05, 0.9, 0.9, {"enabled": False})[1] == 1.0   # A1
    assert pcs_score(0.5, 0.2, 0.3, {"use_T": False, "use_R": False}) == 0.5  # A2 only-G


def test_negatives():
    rng = np.random.default_rng(0)
    claims = [{"statement": f"s{i}", "polarity": "+", "evidence": [1]} for i in range(4)]
    flipped = n2_polarity_flip(claims, rng)
    assert sum(c["polarity"] == "-" for c in flipped) == 2 and all(c["polarity"] == "+" for c in claims)
    recs = [{"G": 0.0}, {"G": 0.7}, {"G": 0.9}, {"G": 0.0}]
    n3 = n3_broken_evidence(claims, recs, {1, 2, 3}, 50, rng)
    assert len(n3) == 2 + 4 and all(e not in {1, 2, 3} for c in n3 if c.get("_n3_random_evidence") for e in c["evidence"])
    vecs = {0: np.array([1.0, 0]), 1: np.array([0.9, 0.1]), 2: np.array([-1.0, 0])}
    assert n1_cross_user(0, [0, 1, 2], vecs) == 2


def test_teacher_distribution():
    r = [[3, 1, 2], [3, 2, 1], [1, 3, 2]]
    assert borda(r, [1, 2, 3])[0] == 3
    q = teacher_distribution(r, [1, 2, 3], tau_T=1.0)
    assert abs(sum(q.values()) - 1) < 1e-9 and q[3] == max(q.values())
