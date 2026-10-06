"""Inconsistent negative profiles (plan §2.3):
  N1 cross-user swap: the profile of another user with the lowest support overlap;
  N2 polarity flip: flip the polarity of a random half of the claims;
  N3 broken evidence: claims with G(c_k) = 0 plus claims whose evidence is replaced by random
     non-history items (statement kept, so the claim is no longer grounded)."""

from __future__ import annotations

import copy

import numpy as np


def n1_cross_user(user: int, users: list[int], profile_vecs: dict[int, np.ndarray]) -> int | None:
    """Return the other user whose (mean-claim) profile vector is least similar to ``user``'s."""
    if user not in profile_vecs:
        return None
    me = profile_vecs[user]
    best, best_sim = None, None
    for v in users:
        if v == user or v not in profile_vecs:
            continue
        sim = float(me @ profile_vecs[v])
        if best_sim is None or sim < best_sim:
            best, best_sim = v, sim
    return best


def n2_polarity_flip(claims: list[dict], rng: np.random.Generator) -> list[dict]:
    out = copy.deepcopy(claims)
    if not out:
        return out
    k = max(1, len(out) // 2)
    for i in rng.choice(len(out), size=k, replace=False):
        out[i]["polarity"] = "-" if out[i]["polarity"] == "+" else "+"
    return out


def n3_broken_evidence(claims: list[dict], claim_recs: list[dict], history: set[int], n_items: int,
                       rng: np.random.Generator) -> list[dict]:
    out = [copy.deepcopy(c) for c, r in zip(claims, claim_recs) if r["G"] == 0.0]
    for c in claims:
        cc = copy.deepcopy(c)
        new_ev = []
        while len(new_ev) < max(1, len(c.get("evidence", []))):
            x = int(rng.integers(n_items))
            if x not in history:
                new_ev.append(x)
        cc["evidence"] = new_ev
        cc["_n3_random_evidence"] = True
        out.append(cc)
    return out
