"""Preference-contrastive representation alignment L_pref (plan §2.4).

z_u = AttnPool({phi(c_k)}) over frozen, polarity-signed claim embeddings (learnable query);
L_pref = - w_u log exp(<g(h_u), z_u>/tau) / (exp(<g(h_u), z_u>/tau) + sum_{z- in {z~_u} ∪ Z_batch} exp(<g(h_u), z->/tau))
z~_u: pooled inconsistent negative profiles N1-N3; Z_batch: other users' profiles (in-batch).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(dim))
        self.key = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.key.weight)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x [B, K, D], mask [B, K] (True = real claim) -> [B, D] (L2-normalised)."""
        att = (self.key(x) * self.query).sum(-1)
        att = att.masked_fill(~mask, -1e4)
        a = torch.softmax(att, dim=1) * mask
        a = a / a.sum(1, keepdim=True).clamp(min=1e-8)
        return F.normalize((a.unsqueeze(-1) * x).sum(1), dim=-1)


def pref_contrast_loss(hu: torch.Tensor, z_pos: torch.Tensor, z_hard: torch.Tensor | None,
                       hard_mask: torch.Tensor | None, w: torch.Tensor, tau: float = 0.1,
                       in_batch: bool = True) -> torch.Tensor:
    """hu [B, D] = normalised g(h_u); z_pos [B, D]; z_hard [B, H, D] with hard_mask [B, H]."""
    pos = (hu * z_pos).sum(-1, keepdim=True) / tau  # [B,1]
    logits = [pos]
    if in_batch and hu.shape[0] > 1:
        sim = hu @ z_pos.t() / tau  # [B,B]
        eye = torch.eye(hu.shape[0], dtype=torch.bool, device=hu.device)
        logits.append(sim.masked_fill(eye, float("-inf")))
    if z_hard is not None and z_hard.shape[1] > 0:
        hs = torch.einsum("bd,bhd->bh", hu, z_hard) / tau
        if hard_mask is not None:
            hs = hs.masked_fill(~hard_mask, float("-inf"))
        logits.append(hs)
    lg = torch.cat(logits, dim=1)
    loss = -(pos.squeeze(1) - torch.logsumexp(lg, dim=1))
    return (w * loss).sum() / w.sum().clamp(min=1e-8)
