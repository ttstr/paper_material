"""Full-catalog softmax cross-entropy for next-item prediction."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def full_softmax_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits: [B, n_items]
    targets: [B] long
    """
    return F.cross_entropy(logits, targets)
