Ablation overrides (deep-merged onto configs/method/pcdrec.yaml by `--ablation`):
A1 Naive-Distill (no PCS, w_u = 1) · A2 PCS single component (G/T/R) · A3 no hard negatives ·
A4 no L_align / no L_pref · A5 ListKL / BPR / no reference model · A6 feature injection
(A6 is only partially wired: the inject head exists but injected inference is not used by the shared evaluator yet).
