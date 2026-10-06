"""G: evidence grounding of profile claims.

G(c_k) = 1[ E_k non-empty and E_k ⊆ allowed history ] * mean_{e in E_k} NLI_entail(x_e, statement_k)
G_u = mean_k G(c_k). Claims citing ids outside the allowed history (probes, valid/test items,
unknown ids) score 0 and are counted as hallucinated-evidence claims.
"""

from __future__ import annotations

import numpy as np


def claim_grounding(claims: list[dict], allowed_history: list[int], item_text: dict, nli) -> list[dict]:
    allowed = set(int(i) for i in allowed_history)
    out, pairs, owners = [], [], []
    for k, c in enumerate(claims):
        ev = [int(e) for e in c.get("evidence", [])]
        valid_ids = bool(ev) and all(e in allowed for e in ev)
        rec = {"k": k, "n_evidence": len(ev), "evidence_valid": valid_ids,
               "hallucinated_evidence": bool(ev) and not valid_ids, "no_evidence": not ev, "entail": [], "G": 0.0}
        out.append(rec)
        if valid_ids:
            for e in ev:
                pairs.append((item_text.get(e, ""), c["statement"]))
                owners.append(k)
    if pairs:
        probs = nli.entail(pairs)
        for k, p in zip(owners, probs):
            out[k]["entail"].append(float(p))
    for r in out:
        if r["evidence_valid"] and r["entail"]:
            r["G"] = float(np.mean(r["entail"]))
    return out


def user_grounding(claim_recs: list[dict]) -> dict:
    if not claim_recs:
        return {"G": 0.0, "n_claims": 0, "halluc_rate": 0.0, "no_evidence_rate": 0.0, "grounded_rate": 0.0}
    G = float(np.mean([r["G"] for r in claim_recs]))
    return {
        "G": G,
        "n_claims": len(claim_recs),
        "halluc_rate": float(np.mean([r["hallucinated_evidence"] for r in claim_recs])),
        "no_evidence_rate": float(np.mean([r["no_evidence"] for r in claim_recs])),
        "grounded_rate": float(np.mean([r["G"] >= 0.5 for r in claim_recs])),
    }
