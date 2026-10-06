"""S-DPO-style multi-negative alignment loss with a frozen reference model (plan §2.4).

L_align = - w_u * log sigma( - log sum_{j- in N_u} exp( beta * [Delta(u, j-) - Delta(u, j+)] ) ),
Delta(u, j) = s_theta(u, j) - s_ref(u, j)   (use_ref=False -> Delta = s_theta, ablation A5).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sdpo_align_loss(s_pos: torch.Tensor, s_neg: torch.Tensor, ref_pos: torch.Tensor | None,
                    ref_neg: torch.Tensor | None, w: torch.Tensor, beta: float = 1.0,
                    neg_mask: torch.Tensor | None = None) -> torch.Tensor:
    """s_pos [B], s_neg [B, m] student scores; ref_* frozen reference scores (or None); w [B]."""
    d_pos = s_pos - (ref_pos if ref_pos is not None else 0.0)
    d_neg = s_neg - (ref_neg if ref_neg is not None else 0.0)
    z = beta * (d_neg - d_pos.unsqueeze(1))
    if neg_mask is not None:
        z = z.masked_fill(~neg_mask, float("-inf"))
    lse = torch.logsumexp(z, dim=1)
    loss = -F.logsigmoid(-lse)
    return (w * loss).sum() / w.sum().clamp(min=1e-8)


def bpr_distill_loss(s_pos, s_neg, w, neg_mask=None):
    """Pairwise BPR distillation variant (A5): mean_j- -log sigma(s+ - s-)."""
    l = -F.logsigmoid(s_pos.unsqueeze(1) - s_neg)
    if neg_mask is not None:
        l = (l * neg_mask).sum(1) / neg_mask.sum(1).clamp(min=1)
    else:
        l = l.mean(1)
    return (w * l).sum() / w.sum().clamp(min=1e-8)
