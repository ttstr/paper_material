# Rebuttal Preparation — PCDRec

**Format constraint (RecSys 2026 rules; re-check for 2027):** one anonymous rebuttal of **500 characters**, only for factual errors or misconceptions; no new results or revision promises. So the answers below must be **in the paper** before submission. The rebuttal can only point to where they are.

| # | Likely reviewer question | Where the paper must answer it | Status |
|---|---|---|---|
| 1 | "Offline LLM → light model is already done (Persona4Rec, Ocean4Rec). What is new?" | Intro (explicit non-claim), Table 1 positioning, Related Work | Written |
| 2 | "Is the gain from text rather than from LLM preferences?" | SASRec-T baseline sharing the same encoder; ablation A8 | TODO (experiments) |
| 3 | "Does verification matter, or is this just distillation?" | Naive-Distill vs. PCDRec; A1–A3; Fig. 2 (gain by PCS bucket) | TODO |
| 4 | "Are the LLM baselines unfairly weak?" | Fairness controls: same teacher, candidates, text, splits, CE loss, tuning budget | Protocol written; runs TODO |
| 5 | "Is NLI grounding reliable on short product titles?" | Human-annotation calibration (Cohen's κ) or an explicit limitation | TODO / optional |
| 6 | "Leakage: does the LLM see future items?" | Temporal hold-out, unit test `test_no_leakage.py` | Implemented |
| 7 | "Online cost really zero?" | Same SASRec graph, `test_online_no_llm.py`, latency table | Test done; latency TODO |
| 8 | "Offline cost is huge." | Cost table (calls, tokens, GPU-h); coverage ablation A9 | TODO |
| 9 | "Only one dataset." | Add Sports/Toys or an Amazon 2023 temporal split if data allow; otherwise state it as a limitation | TODO |
| 10 | "Your SASRec numbers differ from paper X." | Full-catalog ranking, no history filtering, CE loss. Baseline is in the range of public SASRec-CE reports (sanity check in the repo README); cite CE-vs-BCE papers | Baseline done (N@10 0.0410±0.0004) |
| 11 | "Sampled metrics vs. full ranking?" | Full ranking is primary; sampled-100 only in the appendix, with a bias note | TODO |
| 12 | "Why S-DPO-style instead of plain KD?" | A5: S-DPO vs. ListKL vs. BPR, with/without θ_ref | TODO |
| 13 | "Teacher position bias?" | R_u term; Kendall-τ analysis | TODO |
| 14 | "Privacy of sending user histories to an LLM?" | Local open-weight model, metadata-only prompts (DATA_STATEMENT §5) | Written |
| 15 | "Reproducibility of LLM outputs?" | Temperature 0, cached artifacts, prompt hashes, released schema and examples | Planned |

500-character rebuttal skeleton (fill at review time):
> We thank the reviewers. Factual clarifications: (1) R2 states X; Sec. Y / Table Z shows ... (2) R3 says the LLM is used online; Sec. 4.4 and Table 3 state 0 LLM calls/request (unit-tested). (3) ...
