"""Compute PCS (G, T, R, F), weights, teacher distributions and negatives N1-N3 from LLM artifacts.

  python -m pcdrec.verify.run_verify --llm-dir results/pilot/llm_qwen1.5b --config configs/method/pcdrec.yaml

Writes <llm-dir>/verify.jsonl (per-user details), <llm-dir>/verify_summary.json, and
<llm-dir>/signals.pt (tensors consumed by pcdrec.distill.train_student; gitignored).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from pcdrec.data.load_splits import load_data_config, load_processed
from pcdrec.encode.text_encoder import encode_sentences, load_or_build
from pcdrec.paths import REPO_ROOT, resolve
from pcdrec.verify.faithfulness import rationale_faithfulness
from pcdrec.verify.grounding import claim_grounding, user_grounding
from pcdrec.verify.negatives import n1_cross_user, n2_polarity_flip, n3_broken_evidence
from pcdrec.verify.pcs import user_weight
from pcdrec.verify.permutation import permutation_stability, position_bias
from pcdrec.verify.teacher import borda, teacher_distribution
from pcdrec.verify.temporal import popularity_buckets, sample_bucket_negatives, temporal_consistency


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def signed_claim_emb(claims: list[dict], enc_name: str) -> np.ndarray:
    if not claims:
        return np.zeros((0, 384), np.float32)
    e = encode_sentences([c["statement"] for c in claims], enc_name)
    pol = np.array([1.0 if c["polarity"] == "+" else -1.0 for c in claims], np.float32)
    return e * pol[:, None]


def run(llm_dir: Path, method_cfg: dict, data_cfg_path: str, nli=None) -> dict:
    t_start = time.time()
    data_cfg = load_data_config(data_cfg_path)
    bundle = load_processed(data_cfg["processed_dir"])
    item_text = bundle["item_text"]
    vcfg, pcs_cfg, neg_cfg = method_cfg["verify"], method_cfg["pcs"], method_cfg["neg"]
    enc_name = method_cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2")
    item_emb = load_or_build(data_cfg["processed_dir"], enc_name)
    if nli is None:
        from pcdrec.verify.nli import NLIScorer
        nli = NLIScorer(vcfg.get("nli_model", "cross-encoder/nli-deberta-v3-xsmall"))
    buckets = popularity_buckets(bundle["train_seq"], int(bundle["n_items"]), int(vcfg.get("pop_buckets", 10)))
    rng = np.random.default_rng(int(vcfg.get("seed", 0)))

    inputs = {r["user"]: r for r in read_jsonl(llm_dir / "inputs.jsonl")}
    profiles = {r["user"]: r for r in read_jsonl(llm_dir / "profiles.jsonl")}
    rankings: dict[int, list[dict]] = {}
    for r in read_jsonl(llm_dir / "rankings.jsonl"):
        rankings.setdefault(r["user"], []).append(r)
    users = sorted(u for u in profiles if u in rankings and u in inputs)

    recs, claim_embs = {}, {}
    for u in users:
        inp, prof = inputs[u], profiles[u]
        claims = prof["claims"]
        crec = claim_grounding(claims, inp["history"], item_text, nli)
        g = user_grounding(crec)
        ce = signed_claim_emb(claims, enc_name)
        claim_embs[u] = ce
        hist_all = set(e["item"] for e in bundle["train_seq"][u])
        negs = sample_bucket_negatives(inp["probes"], hist_all | set(inp["forbidden"]), buckets,
                                       int(vcfg.get("n_neg", 30)), rng)
        pol = np.ones(len(claims), np.float32)  # signs already folded into ce
        T = temporal_consistency(ce, pol, inp["probes"], negs, item_emb) if len(claims) else 0.5
        rk = sorted(rankings[u], key=lambda x: x["perm"])
        R = permutation_stability([r["ranking"] for r in rk])
        pb = position_bias([r["presented"] for r in rk], [r["ranking"] for r in rk])
        reasons = [x for r in rk for x in r["reasons"]]
        F = rationale_faithfulness(reasons, len(claims), item_text, nli)
        cands = inp["candidates"]
        q = teacher_distribution([r["ranking"] for r in rk], cands, float(method_cfg["loss"].get("tau_T", 2.0)))
        order = borda([r["ranking"] for r in rk], cands)
        gt = inp["teacher_target"]
        gt_rank = order.index(gt)
        PCS, w = user_weight(g["G"], T, R["R"], pcs_cfg)
        recs[u] = {"user": u, **{k: g[k] for k in g}, "T": T, "R": R["R"], "tau_mean": R["tau_mean"],
                   "position_bias_tau": pb, "F": F["F"], "n_reasons": F["n_reasons"],
                   "reason_cite_valid_rate": F["cite_valid_rate"], "PCS": PCS, "w": w,
                   "teacher_gt_rank": gt_rank, "gt_in_stage0_topM": inp.get("gt_in_stage0_topM"),
                   "parse": {"profile": prof["report"], "rank": [r["report"] for r in rk]},
                   "claims": claims, "claim_grounding": crec, "borda": order, "q_map": q}

    # profile vectors for N1 (mean signed claim embedding)
    pvec = {}
    for u, ce in claim_embs.items():
        if len(ce):
            v = ce.mean(0)
            pvec[u] = v / (np.linalg.norm(v) + 1e-8)

    m_neg = int(method_cfg["loss"].get("m_neg", 5))
    align_rank_max = int(method_cfg["loss"].get("align_gt_max_rank", 10))
    signals = {}
    for u in users:
        r, inp = recs[u], inputs[u]
        claims = r["claims"]
        neg_sets = {}
        if neg_cfg.get("n1", True):
            v = n1_cross_user(u, users, pvec)
            if v is not None:
                neg_sets["n1"] = claim_embs[v]
                r["n1_user"] = v
        if neg_cfg.get("n2", True) and claims:
            neg_sets["n2"] = signed_claim_emb(n2_polarity_flip(claims, rng), enc_name)
        if neg_cfg.get("n3", True) and claims:
            hist = set(e["item"] for e in bundle["train_seq"][u])
            n3 = n3_broken_evidence(claims, r["claim_grounding"], hist, int(bundle["n_items"]), rng)
            # N3 keeps statements but breaks grounding: down-weight their embeddings by their (zero) G,
            # i.e. the negative profile is the ungrounded version of the claim set.
            neg_sets["n3"] = signed_claim_emb(n3, enc_name) * float(neg_cfg.get("n3_scale", 1.0))
        cands = inp["candidates"]
        order = r["borda"]
        pos = inp["teacher_target"] if r["teacher_gt_rank"] < align_rank_max else -1
        neg_items = [c for c in order[::-1] if c != inp["teacher_target"]][:m_neg]
        q = r["q_map"]
        signals[u] = {
            "context": inp["teacher_context"], "cands": torch.tensor(cands), "q": torch.tensor([q[c] for c in cands], dtype=torch.float32),
            "pos": int(pos), "negs": torch.tensor(neg_items), "w": float(r["w"]), "PCS": r["PCS"],
            "claims": torch.from_numpy(claim_embs[u]), **{k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in neg_sets.items()},
        }

    out_rows = []
    for u in users:
        r = dict(recs[u])
        r.pop("borda")
        r.pop("q_map")
        out_rows.append(r)
    with open(llm_dir / "verify.jsonl", "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    torch.save({"signals": signals, "pcs_cfg": pcs_cfg, "neg_cfg": neg_cfg, "nli": nli.model_name,
                "text_encoder": enc_name}, llm_dir / "signals.pt")

    def m(key):
        vals = [r[key] for r in out_rows if r.get(key) is not None and not (isinstance(r[key], float) and np.isnan(r[key]))]
        return {"mean": float(np.mean(vals)) if vals else None, "n": len(vals)}

    summary = {
        "n_users": len(users), "nli_model": nli.model_name, "text_encoder": enc_name,
        "claims_per_user": m("n_claims"), "G": m("G"), "halluc_evidence_rate": m("halluc_rate"),
        "no_evidence_rate": m("no_evidence_rate"), "grounded_claim_rate": m("grounded_rate"), "T_auc": m("T"),
        "R": m("R"), "kendall_tau": m("tau_mean"), "position_bias_tau": m("position_bias_tau"), "F": m("F"),
        "reason_cite_valid_rate": m("reason_cite_valid_rate"), "PCS": m("PCS"), "w": m("w"),
        "users_hard_filtered(w=0)": int(sum(r["w"] == 0.0 for r in out_rows)),
        "teacher_gt_rank_mean(0=top)": m("teacher_gt_rank"),
        "teacher_gt_top1_rate": float(np.mean([r["teacher_gt_rank"] == 0 for r in out_rows])) if out_rows else None,
        "teacher_gt_top5_rate": float(np.mean([r["teacher_gt_rank"] < 5 for r in out_rows])) if out_rows else None,
        "align_eligible_users": int(sum(s["pos"] >= 0 for s in signals.values())),
        "profile_parse_rate": float(np.mean([r["parse"]["profile"]["parsed"] for r in out_rows])) if out_rows else None,
        "profile_salvaged_rate": float(np.mean([r["parse"]["profile"].get("salvaged", False) for r in out_rows])) if out_rows else None,
        "rank_parse_rate": float(np.mean([x["parsed"] for r in out_rows for x in r["parse"]["rank"]])) if out_rows else None,
        "rank_missing_appended_mean": float(np.mean([x["n_missing_appended"] for r in out_rows for x in r["parse"]["rank"]])) if out_rows else None,
        "pcs_cfg": pcs_cfg,
        "elapsed_sec": time.time() - t_start,
    }
    (llm_dir / "verify_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-dir", required=True)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/method/pcdrec.yaml"))
    ap.add_argument("--data-config", default=str(REPO_ROOT / "configs/data/beauty.yaml"))
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    print(json.dumps(run(resolve(args.llm_dir), cfg, args.data_config), indent=2))


if __name__ == "__main__":
    main()
