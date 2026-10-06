# Submission Checklist — PCDRec (ACM RecSys, Main Track, Long Paper)

## 0. Which rules apply (verified 2026-10-06 UTC+8)
- **Source used:** RecSys 2026 Call for Contributions, https://recsys.acm.org/recsys26/call/ (fetched 2026-10-06), and the RecSys 2026 camera-ready page https://recsys.acm.org/recsys26/camera-ready/ (page limits).
- **RecSys 2027 has no CFP yet.** https://recsys.acm.org/recsys27/ only lists the venue and dates (Honolulu, 23–27 Aug 2027). https://recsys.acm.org/recsys27/call/ redirects to an old edition's call. The rules below are the **2026 rules**; re-check everything when the 2027 CFP appears. The 2026 deadlines (abstract 14 Apr, paper 21 Apr 2026, AoE) have passed, so the realistic target is RecSys 2027, whose dates fall about a month earlier than usual. **TODO: confirm the 2027 deadlines.**

## 1. Format (RecSys 2026 rules)
- [x] ACM `acmart`, double-column. The CFP prescribes `\documentclass[sigconf,anonymous]{acmart}`; the draft uses `sigconf,review,anonymous` (`review` only adds line numbers). Do **not** use `manuscript`.
- [ ] Use only packages from ACM's TAPS whitelist. The draft uses `booktabs` and `multirow`. TODO: check both against the current list.
- [ ] Add alt text (`\Description{}`) to every figure (placeholders already have it). Use colour-accessible plots.
- [ ] Before submission remove `\todo`/`\tbd` macros and all red text. Finalize the ACM metadata (`\acmConference`).

## 2. Page limit
- Long paper: **max 8 content pages** in the ACM 2-column template, *including figures, tables, appendices and acknowledgements*; references do not count. Nothing except references may follow page 8 (desk-rejection criterion).
- Current draft: **6 pages total, about 5 of content** with placeholders. Budget roughly 2.5 more pages for real tables, figures and discussion. Put extra material (sampled-100, full hyper-parameter grids) in the **auxiliary/supplementary material**, which reviewers are not required to read, or in the anonymous repo.

## 3. Anonymity (mutually anonymous; violations are desk-rejected)
- [x] No author names or affiliations ("Anonymous Author(s)").
- [x] No acknowledgements of people or funding (the Acks section only holds the GenAI-disclosure TODO).
- [x] Prior own work, if any, cited in the third person (TODO: re-check once the related-work list is final).
- [x] Code link is an anonymous placeholder: `https://anonymous.4open.science/r/PCDRec-XXXX`. **TODO: create the mirror.** Never link the real git remote.
- [ ] When creating the anonymous mirror: export a snapshot **without `.git`**. Make sure no usernames, emails or absolute home paths reveal identity. The current commit authors are `pcdrec <pcdrec@local>` and `pcdrec bot <noreply@example.com>` (neutral). Scrub PDF metadata (author field).
- [ ] Pre-print policy: an anonymous version may be posted to arXiv at any time. A non-anonymous version overlapping the submission must be disclosed in EasyChair and needs a distinct title and abstract. No non-anonymous posting during review.

## 4. Supplementary / auxiliary material
- The CFP encourages auxiliary material for reproducibility and/or a link to an anonymous repository. Reviewers are not obliged to review it, and the paper must be self-contained.
- Planned package: this folder (REPRODUCIBILITY.md, DATA_STATEMENT.md), code snapshot, configs, aggregate result CSV/JSON. **No raw Amazon data** (see DATA_STATEMENT.md).

## 5. Ethics, AI use, policies
- [ ] **Generative-AI disclosure:** the CFP requires disclosing GenAI-generated material in a section titled "Acknowledgments" (light editing excepted). Since this draft was produced with AI drafting assistance, disclose it there (TODO). Only humans may be authors.
- [ ] Human-subjects statement only if the optional claim-annotation study is run (IRB/exemption).
- [ ] Data license/terms statement (DATA_STATEMENT.md §2; TODO: verify Amazon data terms).
- [ ] Declare conflicts of interest with PC/SPC in EasyChair (missing declarations can cause desk rejection). Fix the author list and order by the **abstract** deadline; no changes afterwards. Obtain ORCID IDs.
- [ ] No dual submission (including other RecSys tracks).
- [ ] At least one author attends in person if accepted. Open-access APC applies unless the institution is in ACM Open.

## 6. Rebuttal
- RecSys 2026: one short anonymous rebuttal of **500 characters**, only to point out factual errors or misconceptions; no new material. See REBUTTAL_PREP.md.

## 7. Outstanding TODOs before submission
1. Run the whole pipeline: offline LLM profiling + teacher ranking → PCS + N1–N3 → distillation (L_align, L_pref) → PCDRec, 5 seeds.
2. All baselines (GRU4Rec, BERT4Rec, SASRec-T/UniSRec, KAR, DLLM2Rec, RDRec, Persona4Rec†, Ocean-feat†, Naive-Distill, LLM rerank reference) and significance tests.
3. Consistency/faithfulness table, ablations A1–A10, Figs. 1–3, sensitivity, group analysis, case study.
4. Latency (P50/P95, CPU/GPU) and offline LLM cost table.
5. Abstract/intro/conclusion claims written from the actual results.
6. Resolve the `TODO: verify` bib entries (see the paper's README).
7. Anonymous repo mirror + replace the URL placeholder; raw-data conversion script; dependency lock file.
8. GenAI disclosure in Acknowledgments; ACM metadata; remove TODO macros; check the page count (≤ 8 content pages).
9. Re-verify all rules against the RecSys 2027 CFP once it is published.
