"""PCDRec student = unmodified SASRec backbone + training-only heads (dropped at export).

- g: projection head h_u -> claim-embedding space (L_pref)
- pool: AttnPool over frozen claim embeddings (L_pref)
- inject (ablation A6 only): h_u + W z_u at inference -> needs cached LLM profiles online
  (violates the zero-dependency serving property; exists only to measure distill vs inject).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pcdrec.losses.pref_contrast import AttnPool
from pcdrec.models.sasrec import SASRec


class PCDRecStudent(nn.Module):
    def __init__(self, backbone: SASRec, claim_dim: int = 384, inject: bool = False):
        super().__init__()
        self.backbone = backbone
        d = backbone.hidden_size
        self.g = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, claim_dim))
        self.pool = AttnPool(claim_dim)
        self.inject = nn.Linear(claim_dim, d, bias=False) if inject else None
        if self.inject is not None:
            nn.init.zeros_(self.inject.weight)

    # the online path: identical to SASRec
    @property
    def pad_id(self):
        return self.backbone.pad_id

    def score(self, users, item_seq):
        return self.backbone.score(users, item_seq)

    def proj(self, h: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.g(h), dim=-1)

    def online_state_dict(self) -> dict:
        return self.backbone.state_dict()
