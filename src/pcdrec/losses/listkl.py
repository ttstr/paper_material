"""ListKL distillation variant: w_u * KL(q_u || softmax_{C_u}(s_theta / tau_S)) (DLLM2Rec-style)."""

from __future__ import annotations

import torch


def listkl_loss(s_cand: torch.Tensor, q: torch.Tensor, w: torch.Tensor, tau_S: float = 1.0) -> torch.Tensor:
    """s_cand [B, M] student scores on candidates, q [B, M] teacher distribution (rows sum to 1)."""
    logp = torch.log_softmax(s_cand / tau_S, dim=1)
    kl = (q * (torch.log(q.clamp(min=1e-12)) - logp)).sum(1)
    return (w * kl).sum() / w.sum().clamp(min=1e-8)
