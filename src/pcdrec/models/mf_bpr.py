"""MF-BPR (Rendle et al., UAI 2009): non-sequential reference baseline.

score(u, i) = p_u . q_i, trained with BPR on (u, i, j) triples where j is a
uniformly sampled item the user has not interacted with in *train*. Uses only
the user id at inference (the valid item is not folded in at test time).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MFBPR(nn.Module):
    def __init__(self, n_users: int, n_items: int, hidden_size: int = 64, initializer_range: float = 0.1):
        super().__init__()
        self.n_users, self.n_items, self.hidden_size = n_users, n_items, hidden_size
        self.pad_id = n_items  # for evaluator compatibility (sequences are ignored)
        self.max_seq_length = 1
        self.user_emb = nn.Embedding(n_users, hidden_size)
        self.item_emb = nn.Embedding(n_items, hidden_size)
        nn.init.normal_(self.user_emb.weight, std=initializer_range)
        nn.init.normal_(self.item_emb.weight, std=initializer_range)

    def bpr_loss(self, u, i, j, l2: float = 0.0):
        pu, qi, qj = self.user_emb(u), self.item_emb(i), self.item_emb(j)
        diff = (pu * qi).sum(-1) - (pu * qj).sum(-1)
        loss = F.softplus(-diff).mean()
        if l2 > 0:
            loss = loss + 0.5 * l2 * (pu.pow(2).sum(-1) + qi.pow(2).sum(-1) + qj.pow(2).sum(-1)).mean()
        return loss

    def score(self, users: torch.Tensor, item_seq=None) -> torch.Tensor:
        return self.user_emb(users) @ self.item_emb.weight.t()

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters())
