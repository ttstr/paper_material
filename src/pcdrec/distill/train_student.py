"""Train the PCDRec student: L = L_rec + lambda1 * L_align + lambda2 * L_pref (plan §2.4).

  python -m pcdrec.distill.train_student --config configs/method/pcdrec.yaml \
      --llm-dir results/pilot/llm_qwen1.5b --tag pcdrec_pilot_s42

Every component is a config switch (ablations A1-A6): pcs.enabled / use_G / use_T / use_R,
neg.n1/n2/n3, loss.align_type (sdpo|listkl|bpr|none), loss.use_ref, loss.lambda1/lambda2,
init_from_ref, inject.enabled. Evaluation uses the shared evaluator; auxiliary heads are
dropped at export (pcdrec.export_online.export_student).
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from pcdrec.data.load_splits import load_data_config, load_processed
from pcdrec.distill.student import PCDRecStudent
from pcdrec.evaluator import evaluate_split, pad_history
from pcdrec.losses.listkl import listkl_loss
from pcdrec.losses.pref_contrast import pref_contrast_loss
from pcdrec.losses.sdpo_align import bpr_distill_loss, sdpo_align_loss
from pcdrec.paths import REPO_ROOT, resolve
from pcdrec.train import SASRecTrainDataset, build_model, parse_sets, rel, set_seed
from pcdrec.verify.pcs import user_weight


def deep_update(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_signals(llm_dir: Path, cfg: dict) -> tuple[list[int], dict]:
    sig = torch.load(llm_dir / "signals.pt", weights_only=False)["signals"]
    ver = {json.loads(l)["user"]: json.loads(l) for l in open(llm_dir / "verify.jsonl", encoding="utf-8")}
    users = sorted(sig)
    for u in users:
        v = ver[u]
        pcs, w = user_weight(v["G"], v["T"], v["R"], cfg["pcs"])  # recomputed from config (A1/A2 switches)
        sig[u]["w"], sig[u]["PCS"] = w, pcs
        hard = [sig[u][k] for k in ("n1", "n2", "n3") if cfg["neg"].get(k, True) and k in sig[u] and len(sig[u][k])]
        sig[u]["hard"] = hard
    return users, sig


def collate_aux(users: list[int], sig: dict, pad_id: int, max_len: int, dim: int):
    B = len(users)
    ctx = torch.stack([pad_history(sig[u]["context"], max_len, pad_id) for u in users])
    cands = torch.stack([sig[u]["cands"] for u in users])
    q = torch.stack([sig[u]["q"] for u in users])
    pos = torch.tensor([sig[u]["pos"] for u in users])
    m = max(len(sig[u]["negs"]) for u in users)
    negs = torch.full((B, m), 0, dtype=torch.long)
    neg_mask = torch.zeros(B, m, dtype=torch.bool)
    for b, u in enumerate(users):
        n = sig[u]["negs"]
        negs[b, : len(n)] = n
        neg_mask[b, : len(n)] = True
    w = torch.tensor([sig[u]["w"] for u in users], dtype=torch.float32)
    K = max(1, max(len(sig[u]["claims"]) for u in users))
    claims = torch.zeros(B, K, dim)
    cmask = torch.zeros(B, K, dtype=torch.bool)
    for b, u in enumerate(users):
        c = sig[u]["claims"]
        claims[b, : len(c)] = c
        cmask[b, : len(c)] = True
    H = max([len(sig[u]["hard"]) for u in users] + [0])
    KH = max([len(h) for u in users for h in sig[u]["hard"]] + [1])
    hard = torch.zeros(B, H, KH, dim)
    hmask = torch.zeros(B, H, KH, dtype=torch.bool)
    for b, u in enumerate(users):
        for i, h in enumerate(sig[u]["hard"]):
            hard[b, i, : len(h)] = h
            hmask[b, i, : len(h)] = True
    return dict(ctx=ctx, cands=cands, q=q, pos=pos, negs=negs, neg_mask=neg_mask, w=w, claims=claims, cmask=cmask,
                hard=hard, hmask=hmask)


def aux_losses(student: PCDRecStudent, ref, batch: dict, cfg: dict) -> dict:
    lc = cfg["loss"]
    hu = student.backbone.user_representation(batch["ctx"])  # [B, d]
    E = student.backbone.output_item_weights()
    out = {}
    w = batch["w"]
    if float(lc.get("lambda1", 0)) > 0 and lc.get("align_type", "sdpo") != "none":
        at = lc.get("align_type", "sdpo")
        if at == "listkl":
            s_c = (hu.unsqueeze(1) * E[batch["cands"]]).sum(-1)
            out["align"] = listkl_loss(s_c, batch["q"], w, float(lc.get("tau_S", 1.0)))
        else:
            ok = batch["pos"] >= 0
            if ok.any():
                pos = batch["pos"].clamp(min=0)
                s_pos = (hu * E[pos]).sum(-1)
                s_neg = (hu.unsqueeze(1) * E[batch["negs"]]).sum(-1)
                if at == "bpr":
                    out["align"] = bpr_distill_loss(s_pos[ok], s_neg[ok], w[ok], batch["neg_mask"][ok])
                else:
                    r_pos = r_neg = None
                    if lc.get("use_ref", True):
                        with torch.no_grad():
                            hr = ref.user_representation(batch["ctx"])
                            Er = ref.output_item_weights()
                            r_pos = (hr * Er[pos]).sum(-1)[ok]
                            r_neg = (hr.unsqueeze(1) * Er[batch["negs"]]).sum(-1)[ok]
                    out["align"] = sdpo_align_loss(s_pos[ok], s_neg[ok], r_pos, r_neg, w[ok], float(lc.get("beta", 1.0)),
                                                   batch["neg_mask"][ok])
    if float(lc.get("lambda2", 0)) > 0:
        z = student.pool(batch["claims"], batch["cmask"])
        g = student.proj(hu)
        zh, hm = None, None
        if batch["hard"].shape[1] > 0:
            B, H, K, D = batch["hard"].shape
            zh = student.pool(batch["hard"].view(B * H, K, D), batch["hmask"].view(B * H, K)).view(B, H, D)
            hm = batch["hmask"].any(-1)
        out["pref"] = pref_contrast_loss(g, z, zh, hm, w, float(lc.get("tau", 0.1)), bool(cfg["neg"].get("in_batch", True)))
    return out


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/method/pcdrec.yaml"))
    ap.add_argument("--data-config", default=str(REPO_ROOT / "configs/data/beauty.yaml"))
    ap.add_argument("--llm-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--set", action="append", default=None)
    ap.add_argument("--ablation", action="append", default=None, help="configs/ablation/*.yaml override(s)")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    for ab in args.ablation or []:
        cfg = deep_update(cfg, yaml.safe_load(open(resolve(ab), encoding="utf-8")))
    cfg = deep_update(cfg, parse_sets(args.set))
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.threads or cfg.get("threads"):
        torch.set_num_threads(int(args.threads or cfg["threads"]))
    set_seed(int(cfg["seed"]))
    data_cfg = load_data_config(args.data_config)
    bundle = load_processed(data_cfg["processed_dir"])
    n_items = int(bundle["n_items"])

    ref_ck = torch.load(resolve(cfg["ref_checkpoint"]), map_location="cpu", weights_only=False)
    mcfg = dict(ref_ck["cfg"])
    mcfg.setdefault("model", "sasrec")
    max_len = int(ref_ck["max_len"])
    ref = build_model(mcfg, n_items, 1, max_len, data_cfg)
    ref.load_state_dict(ref_ck["model_state"])
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    backbone = build_model(mcfg, n_items, 1, max_len, data_cfg)
    if cfg.get("init_from_ref", True):
        backbone.load_state_dict(ref_ck["model_state"])
    llm_dir = resolve(args.llm_dir)
    t_users, sig = load_signals(llm_dir, cfg)
    dim = int(next(iter(sig.values()))["claims"].shape[1]) if sig else 384
    student = PCDRecStudent(backbone, claim_dim=dim, inject=bool(cfg.get("inject", {}).get("enabled", False)))

    oc = cfg["optim"]
    ds = SASRecTrainDataset(bundle["train_seq"], max_len=max_len, pad_id=backbone.pad_id)
    loader = DataLoader(ds, batch_size=int(oc["batch_size"]), shuffle=True)
    opt = torch.optim.Adam(student.parameters(), lr=float(oc["lr"]))
    lc = cfg["loss"]
    l1, l2 = float(lc.get("lambda1", 0)), float(lc.get("lambda2", 0))
    aux_bs = int(cfg.get("aux_batch_size", 32))
    aux_every = int(cfg.get("aux_every", 1))
    rng = random.Random(int(cfg["seed"]))

    results_dir = resolve(args.results_dir or cfg.get("results_dir", "results"))
    ck_dir = resolve(cfg.get("checkpoint_dir", "results/checkpoints"))
    for d in (results_dir, ck_dir, results_dir / "per_user"):
        d.mkdir(parents=True, exist_ok=True)
    best_path = ck_dir / f"{args.tag}_best.pt"

    def evaluate(split: str, ranks=False):
        if split == "valid":
            return evaluate_split(student, bundle["train_seq"], bundle["valid_target"], max_len, torch.device("cpu"),
                                  history_mode="train", return_ranks=ranks)
        return evaluate_split(student, bundle["train_seq"], bundle["test_target"], max_len, torch.device("cpu"),
                              history_mode="train_valid", valid_target=bundle["valid_target"], return_ranks=ranks)

    v0 = evaluate("valid")
    print(f"epoch=0 (init{'=ref' if cfg.get('init_from_ref', True) else '=scratch'}) valid_ndcg@10={v0['ndcg@10']:.6f}", flush=True)
    best, bad, hist, t0 = v0["ndcg@10"], 0, [], time.time()
    torch.save({"student_state": student.state_dict(), "online_state": student.online_state_dict(), "cfg": cfg,
                "model_cfg": mcfg, "n_items": n_items, "max_len": max_len, "epoch": 0}, best_path)
    step = 0
    for ep in range(1, int(oc["epochs"]) + 1):
        student.train()
        te = time.time()
        sums = {"rec": 0.0, "align": 0.0, "pref": 0.0}
        n_aux = 0
        for seq_in, seq_tgt in loader:
            h = student.backbone(seq_in)
            vp = seq_tgt.ne(-100)
            loss_rec = F.cross_entropy(h[vp] @ student.backbone.output_item_weights().t(), seq_tgt[vp])
            loss = loss_rec
            sums["rec"] += loss_rec.item()
            if t_users and (l1 > 0 or l2 > 0) and step % aux_every == 0:
                bu = rng.sample(t_users, min(aux_bs, len(t_users)))
                bu = [u for u in bu if sig[u]["w"] > 0] or bu
                ax = aux_losses(student, ref, collate_aux(bu, sig, backbone.pad_id, max_len, dim), cfg)
                if "align" in ax:
                    loss = loss + l1 * ax["align"]
                    sums["align"] += ax["align"].item()
                if "pref" in ax:
                    loss = loss + l2 * ax["pref"]
                    sums["pref"] += ax["pref"].item()
                n_aux += 1
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
        v = evaluate("valid")
        row = {"epoch": ep, "loss_rec": sums["rec"] / len(loader), "loss_align": sums["align"] / max(n_aux, 1),
               "loss_pref": sums["pref"] / max(n_aux, 1), "aux_steps": n_aux, "epoch_sec": time.time() - te,
               **{f"valid_{k}": x for k, x in v.items()}}
        hist.append(row)
        print(f"epoch={ep} rec={row['loss_rec']:.4f} align={row['loss_align']:.4f} pref={row['loss_pref']:.4f} "
              f"valid_ndcg@10={v['ndcg@10']:.6f} sec={row['epoch_sec']:.0f}", flush=True)
        if v["ndcg@10"] > best:
            best, bad = v["ndcg@10"], 0
            torch.save({"student_state": student.state_dict(), "online_state": student.online_state_dict(),
                        "cfg": cfg, "model_cfg": mcfg, "n_items": n_items, "max_len": max_len, "epoch": ep}, best_path)
        else:
            bad += 1
            if bad >= int(oc["patience"]):
                break
    ck = torch.load(best_path, weights_only=False)
    student.load_state_dict(ck["student_state"])
    vm, uv, rv = evaluate("valid", True)
    tm, ut, rt = evaluate("test", True)
    np.savez_compressed(results_dir / "per_user" / f"{args.tag}.npz", users=ut.astype(np.int32),
                        valid_rank=rv.astype(np.int32), test_rank=rt.astype(np.int32))
    # metrics on the teacher-covered users only (pilot diagnostics)
    tu = np.isin(ut, np.asarray(t_users))
    sub = {}
    for name, r in (("valid", rv), ("test", rt)):
        rr = r[tu]
        sub[name] = {"n": int(tu.sum()), "ndcg@10": float(np.where(rr <= 10, 1 / np.log2(rr + 1), 0).mean()) if tu.any() else None,
                     "hr@10": float((rr <= 10).mean()) if tu.any() else None}
    out = {"tag": args.tag, "method": "pcdrec_student", "llm_dir": rel(llm_dir), "n_teacher_users": len(t_users),
           "best_epoch": ck["epoch"], "epochs_run": len(hist), "valid": vm, "test": tm, "teacher_users_subset": sub,
           "init_valid_ndcg@10": v0["ndcg@10"], "elapsed_sec": time.time() - t0, "seed": cfg["seed"],
           "n_params_online": sum(p.numel() for p in student.backbone.parameters()),
           "n_params_train_total": sum(p.numel() for p in student.parameters()), "config": cfg, "history": hist,
           "checkpoint": rel(best_path), "note": "written by pcdrec.distill.train_student; do not hand-edit"}
    (results_dir / f"{args.tag}_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("tag", "valid", "test", "teacher_users_subset", "best_epoch")}, indent=2))
    return out


if __name__ == "__main__":
    main()
