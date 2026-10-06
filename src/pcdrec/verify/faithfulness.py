"""F: rationale faithfulness. A reason r_{u,j} is faithful iff every cited claim id exists in P_u and
the item text entails the reason: F_{u,j} = 1[cites ⊆ claims, non-empty] * NLI_entail(x_j, r_{u,j})."""

from __future__ import annotations

import numpy as np


def rationale_faithfulness(reasons: list[dict], n_claims: int, item_text: dict, nli) -> dict:
    if not reasons:
        return {"F": None, "n_reasons": 0, "cite_valid_rate": None, "per_item": []}
    pairs, idx, per = [], [], []
    for r in reasons:
        cl = r.get("claims", [])
        ok = bool(cl) and all(0 <= int(c) < n_claims for c in cl)
        per.append({"item": r["item"], "cite_valid": ok, "entail": 0.0, "F": 0.0})
        if ok and r.get("reason"):
            pairs.append((item_text.get(int(r["item"]), ""), r["reason"]))
            idx.append(len(per) - 1)
    if pairs:
        for i, p in zip(idx, nli.entail(pairs)):
            per[i]["entail"] = float(p)
            per[i]["F"] = float(p)
    return {"F": float(np.mean([p["F"] for p in per])), "n_reasons": len(per),
            "cite_valid_rate": float(np.mean([p["cite_valid"] for p in per])), "per_item": per}
