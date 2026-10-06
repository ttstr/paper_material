#!/usr/bin/env python
"""Summarise the CPU pilot (LLM ledger, PCS verification, student distillation, online export) and estimate the
cost of running the offline LLM stages for ALL users. Every measured number is read from files written by the
pipeline; every non-measured number is an explicit, sourced ASSUMPTION listed in ASSUMPTIONS below.

  python scripts/pilot_report.py --llm-dir results/pilot/llm_qwen1.5b --out results/pilot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
N_USERS_FULL = 22363

# Non-measured inputs for the GPU / API estimates. Change them here; each one carries its source.
ASSUMPTIONS = {
    "gpu_decode_tok_s_batched": {
        "value": 980.0,
        "what": "Qwen2.5-7B-Instruct FP16, vLLM, single RTX 4090 24GB, aggregate decode tokens/s at batch 32 (prompt 512, gen 256)",
        "source": "https://gigagpu.com/rtx-4090-24gb-for-qwen-25-7b/ (third-party benchmark, vLLM 0.6.3, FP8 KV cache; accessed 2026-10-06)"},
    "gpu_decode_tok_s_batch1": {
        "value": 105.0,
        "what": "Qwen2.5-7B-Instruct FP16, vLLM, single RTX 4090 24GB, batch-1 decode tokens/s (pessimistic: no batching)",
        "source": "https://gigagpu.com/rtx-4090-24gb-for-qwen-25-7b/ (accessed 2026-10-06); cross-check: the official Qwen2.5 "
                  "speed benchmark reports 84.28 tok/s for 7B BF16 vLLM batch 1 on an A100 80GB "
                  "(https://qwen.readthedocs.io/en/v2.5/benchmark/speed_benchmark.html)"},
    "gpu_prefill_tok_s": {
        "value": 4000.0,
        "what": "Qwen2.5-7B prefill tokens/s on RTX 4090 (reported 3,800-4,200 t/s at 8k prompt for AWQ/FP8)",
        "source": "https://gigagpu.com/rtx-4090-24gb-for-qwen-25-7b/ (accessed 2026-10-06)"},
    "api_usd_per_M_input": {
        "value": 0.10,
        "what": "qwen/qwen-2.5-7b-instruct list price on OpenRouter, input",
        "source": "https://openrouter.ai/qwen/qwen-2.5-7b-instruct (list price as observed 2026-09/10; may change)"},
    "api_usd_per_M_output": {
        "value": 0.20,
        "what": "qwen/qwen-2.5-7b-instruct list price on OpenRouter, output",
        "source": "https://openrouter.ai/qwen/qwen-2.5-7b-instruct (list price as observed 2026-09/10; may change)"},
    "same_token_counts_for_7b": {
        "value": True,
        "what": "prompt token counts transfer exactly (Qwen2.5 1.5B and 7B share one tokenizer); completion lengths of the 7B teacher are ASSUMED equal to the 1.5B pilot's",
        "source": "Qwen2.5 model cards (shared tokenizer); completion-length equality is an assumption"},
}


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]


def mean(xs):
    return float(np.mean(xs)) if len(xs) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-dir", default="results/pilot/llm_qwen1.5b")
    ap.add_argument("--llm-config", default="configs/llm/qwen2.5-1.5b-cpu.yaml")
    ap.add_argument("--full-config", default="configs/llm/qwen2.5-7b.yaml")
    ap.add_argument("--out", default="results/pilot")
    args = ap.parse_args()
    llm_dir, out = ROOT / args.llm_dir, ROOT / args.out
    pcfg = yaml.safe_load(open(ROOT / args.llm_config))
    fcfg = yaml.safe_load(open(ROOT / args.full_config))
    calls = load_jsonl(llm_dir / "calls.jsonl")
    ledger = json.loads((llm_dir / "ledger.json").read_text())
    k_reason = int(pcfg["offline"]["reasons_top_k"])
    reason_perms = set(pcfg["offline"].get("reasons_in_perms", [0]))

    groups = {"profile": [c for c in calls if c["stage"] == "profile"],
              "rank+reasons": [c for c in calls if c["stage"] == "rank" and c.get("perm") in reason_perms],
              "rank": [c for c in calls if c["stage"] == "rank" and c.get("perm") not in reason_perms]}
    meas = {}
    for g, cs in groups.items():
        meas[g] = {"calls": len(cs), "prompt_tok": mean([c["prompt_tokens"] for c in cs]),
                   "completion_tok": mean([c["completion_tokens"] for c in cs]),
                   "sec_per_call": mean([c["seconds"] for c in cs]),
                   "parse_ok_rate": mean([float(c["parse_ok"]) for c in cs]),
                   "hit_max_tokens_rate": None}
    n_users_llm = len({c["user"] for c in groups["profile"]})
    total_sec = sum(c["seconds"] for c in calls)
    users_complete = len({c["user"] for c in groups["rank"] + groups["rank+reasons"]})
    sec_per_user = total_sec / max(n_users_llm, 1)
    comp_tok_s = sum(c["completion_tokens"] for c in calls) / total_sec
    o_reason = max(meas["rank+reasons"]["completion_tok"] - meas["rank"]["completion_tok"], 0.0) / k_reason

    P_p = int(pcfg["offline"]["n_perm"])
    P_f = int(fcfg["offline"]["n_perm"])
    kf = int(fcfg["offline"]["reasons_top_k"])
    nrp_f = len(fcfg["offline"].get("reasons_in_perms", [0]))
    # per-user tokens under (a) the pilot config, (b) the full plan config (reasons for all M candidates, all perms)
    per_user = {
        "pilot_config": {
            "prompt_tok": meas["profile"]["prompt_tok"] + len(reason_perms) * meas["rank+reasons"]["prompt_tok"]
                          + (P_p - len(reason_perms)) * meas["rank"]["prompt_tok"],
            "completion_tok": meas["profile"]["completion_tok"] + len(reason_perms) * meas["rank+reasons"]["completion_tok"]
                              + (P_p - len(reason_perms)) * meas["rank"]["completion_tok"],
            "calls": 1 + P_p},
        "full_plan_config": {
            "prompt_tok": meas["profile"]["prompt_tok"] + P_f * meas["rank+reasons"]["prompt_tok"],
            "completion_tok": meas["profile"]["completion_tok"] + nrp_f * (meas["rank"]["completion_tok"] + kf * o_reason)
                              + (P_f - nrp_f) * meas["rank"]["completion_tok"],
            "calls": 1 + P_f},
    }
    A = {k: v["value"] for k, v in ASSUMPTIONS.items()}
    est = {}
    for name, pu in per_user.items():
        Ptot, Otot = N_USERS_FULL * pu["prompt_tok"], N_USERS_FULL * pu["completion_tok"]
        cpu_h = (N_USERS_FULL * sec_per_user / 3600) if name == "pilot_config" else (Otot / comp_tok_s / 3600)
        est[name] = {
            "per_user": pu, "total_calls": N_USERS_FULL * pu["calls"], "total_prompt_tok": Ptot, "total_completion_tok": Otot,
            "cpu_1.5b_hours(measured_rate)": cpu_h,
            "gpu_7b_hours_batched(assumed)": (Otot / A["gpu_decode_tok_s_batched"] + Ptot / A["gpu_prefill_tok_s"]) / 3600,
            "gpu_7b_hours_batch1(assumed)": (Otot / A["gpu_decode_tok_s_batch1"] + Ptot / A["gpu_prefill_tok_s"]) / 3600,
            "api_usd(assumed_price)": Ptot / 1e6 * A["api_usd_per_M_input"] + Otot / 1e6 * A["api_usd_per_M_output"],
        }

    vs = json.loads((llm_dir / "verify_summary.json").read_text()) if (llm_dir / "verify_summary.json").exists() else None
    students = {}
    for p in sorted(out.glob("*_metrics.json")):
        d = json.loads(p.read_text())
        students[d["tag"]] = d
    ref = json.loads((ROOT / "results/sasrec_full_s42_metrics.json").read_text())
    export = json.loads((out / "online_export_check.json").read_text()) if (out / "online_export_check.json").exists() else None

    report = {"measured": {"llm": {"model": ledger.get("model"), "backend": ledger.get("backend"), "dtype": ledger.get("dtype"),
                                   "threads": ledger.get("threads"), "batch_size": ledger.get("batch_size"),
                                   "n_users": n_users_llm, "users_with_rankings": users_complete,
                                   "total_wall_sec": total_sec, "sec_per_user": sec_per_user,
                                   "completion_tok_per_sec": comp_tok_s, "by_group": meas,
                                   "reason_tokens_per_candidate": o_reason},
                           "verify": vs, "students": {k: {kk: v.get(kk) for kk in ("valid", "test", "teacher_users_subset", "best_epoch",
                                                                                    "epochs_run", "elapsed_sec", "init_valid_ndcg@10",
                                                                                    "n_params_online", "n_teacher_users")}
                                                      for k, v in students.items()},
                           "reference_sasrec_s42": {"valid": ref["valid"], "test": ref["test"]}, "online_export": export},
              "estimate_full_22363_users": est, "assumptions": ASSUMPTIONS}
    (out / "pilot_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    L = []
    L.append("# CPU pilot — small model, small user subset (NOT a main-table result)")
    L.append("")
    L.append("Auto-generated by `scripts/pilot_report.py`; do not hand-edit. This pilot only shows that the offline pipeline "
             "(profile → teacher ranking → PCS verification → distillation → online export → evaluation) runs end to end "
             "with a real LLM. The teacher is **Qwen2.5-1.5B-Instruct on CPU**, not the paper's Qwen2.5-7B-Instruct, and only "
             f"**{n_users_llm} users** get LLM signals; the student metrics below are therefore not evidence for or against the method.")
    L.append("")
    L.append("## 1. LLM stage (measured)")
    L.append("")
    L.append(f"- model `{ledger.get('model')}`, backend `{ledger.get('backend')}`, dtype {ledger.get('dtype')}, "
             f"{ledger.get('threads')} CPU threads, batch {ledger.get('batch_size')} (greedy); box shared with 2 training jobs × 3 threads.")
    L.append(f"- users: {n_users_llm}; total generation wall time {total_sec / 60:.1f} min → **{sec_per_user:.1f} s/user**; "
             f"aggregate decode {comp_tok_s:.1f} completion tok/s.")
    L.append("")
    L.append("| call type | #calls | prompt tok/call | completion tok/call | s/call (amortised over batch) | JSON parse ok |")
    L.append("|---|---|---|---|---|---|")
    for g, m in meas.items():
        L.append(f"| {g} | {m['calls']} | {m['prompt_tok']:.0f} | {m['completion_tok']:.0f} | {m['sec_per_call']:.1f} | {m['parse_ok_rate']:.2f} |")
    L.append("")
    if vs:
        L.append("## 2. PCS verification (measured, no LLM judge)")
        L.append("")
        keys = ["n_users", "claims_per_user", "G", "halluc_evidence_rate", "grounded_claim_rate", "T_auc", "R", "kendall_tau",
                "position_bias_tau", "F", "PCS", "w", "users_hard_filtered(w=0)", "teacher_gt_rank_mean(0=top)",
                "teacher_gt_top1_rate", "teacher_gt_top5_rate", "align_eligible_users", "profile_parse_rate",
                "profile_salvaged_rate", "rank_parse_rate", "rank_missing_appended_mean", "elapsed_sec"]
        L.append("| quantity | value |")
        L.append("|---|---|")
        for k in keys:
            v = vs.get(k)
            if isinstance(v, dict):
                v = v.get("mean")
            L.append(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")
        L.append(f"\nNLI model `{vs.get('nli_model')}`, sentence encoder `{vs.get('text_encoder')}`; candidates M = "
                 f"{pcfg['offline']['cand_M']} from Stage-0 SASRec (seed 42), so a random teacher puts the target in the top-5 with p = "
                 f"{5 / int(pcfg['offline']['cand_M']):.2f}.")
        L.append("")
    if students:
        L.append("## 3. Student distillation (all 22,363 users evaluated; LLM signals only for the pilot users)")
        L.append("")
        L.append("| run | valid NDCG@10 | test NDCG@10 | test HR@10 | teacher users: test NDCG@10 (n) | best epoch | minutes |")
        L.append("|---|---|---|---|---|---|---|")
        L.append(f"| SASRec seed 42 (Stage-0 / reference) | {ref['valid']['ndcg@10']:.4f} | {ref['test']['ndcg@10']:.4f} | "
                 f"{ref['test']['hr@10']:.4f} | - | - | - |")
        for k, d in students.items():
            sub = d["teacher_users_subset"]["test"]
            L.append(f"| {k} | {d['valid']['ndcg@10']:.4f} | {d['test']['ndcg@10']:.4f} | {d['test']['hr@10']:.4f} | "
                     f"{sub['ndcg@10']:.4f} ({sub['n']}) | {d['best_epoch']} | {d['elapsed_sec'] / 60:.1f} |")
        L.append("")
    if export:
        L.append("## 4. Online export (measured)")
        L.append("")
        for k, v in export.items():
            L.append(f"- {k}: {v}")
        L.append("")
    L.append(f"## 5. Cost estimate for all {N_USERS_FULL:,} users")
    L.append("")
    L.append("Notation per user: c = calls, p = prompt tokens, o = completion tokens (pilot means). "
             "N = 22,363.")
    L.append("")
    L.append("- CPU, 1.5B, pilot config (measured): T = N · s_user, with s_user the measured generation seconds per user.")
    L.append("- CPU, 1.5B, full-plan config: T = N · o_user / r_dec, with r_dec the measured aggregate completion tok/s.")
    L.append("- 7B on one 24 GB GPU (**assumed**): T = N · (o_user / R_dec + p_user / R_prefill).")
    L.append("- API (**assumed price**): cost = N · (p_user · $in + o_user · $out) / 10⁶.")
    L.append(f"- Full-plan config (`{args.full_config}`): reasons for all {kf} candidates in {nrp_f} of {P_f} permutations. "
             f"Its completion length is extrapolated as o_rank + {kf} · o_reason, with o_reason = {o_reason:.1f} tok "
             f"= (o_rank+reasons − o_rank) / {k_reason}, measured in the pilot.")
    L.append("")
    L.append("| config | calls | prompt tok (M) | completion tok (M) | CPU 1.5B h (measured rate) | 7B GPU h, batched (assumed) | 7B GPU h, batch 1 (assumed) | API USD (assumed price) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for name, e in est.items():
        L.append(f"| {name} | {e['total_calls']:,} | {e['total_prompt_tok'] / 1e6:.1f} | {e['total_completion_tok'] / 1e6:.1f} | "
                 f"{e['cpu_1.5b_hours(measured_rate)']:.0f} | {e['gpu_7b_hours_batched(assumed)']:.1f} | "
                 f"{e['gpu_7b_hours_batch1(assumed)']:.1f} | {e['api_usd(assumed_price)']:.2f} |")
    L.append("")
    L.append("Assumptions (not measured on this box):")
    L.append("")
    for k, v in ASSUMPTIONS.items():
        L.append(f"- `{k}` = {v['value']}: {v['what']}. Source: {v['source']}")
    L.append("")
    L.append("PCS verification and distillation costs are not LLM costs. They are measured above: the verify elapsed_sec "
             "and the student minutes, on CPU.")
    (out / "PILOT_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
