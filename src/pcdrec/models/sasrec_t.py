"""SASRec-T: SASRec + frozen sentence-embedding item text features.

Item representation e_j = id_emb(j) + W t_j, where t_j is the cached, frozen,
L2-normalised sentence embedding of the item text (title + categories + brand)
and W is a learnable linear projection (no bias). The same representation is
used for the input sequence and for the tied output (full-catalogue softmax).
Backbone (layers / heads / d / max_len) is identical to SASRec; the only extra
parameters are W (text_dim x d).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .sasrec import SASRec


class SASRecT(SASRec):
    def __init__(self, n_items: int, text_emb: np.ndarray | torch.Tensor, **kw):
        super().__init__(n_items=n_items, **kw)
        t = torch.as_tensor(np.asarray(text_emb), dtype=torch.float32)
        assert t.shape[0] == n_items, (t.shape, n_items)
        pad = torch.zeros(1, t.shape[1])
        self.register_buffer("text_feat", torch.cat([t, pad], 0), persistent=False)
        self.text_proj = nn.Linear(t.shape[1], self.hidden_size, bias=False)
        self.text_proj.weight.data.normal_(0.0, self.initializer_range)

    def _all_item_repr(self) -> torch.Tensor:
        rep = self.item_emb.weight + self.text_proj(self.text_feat)
        # keep the padding row exactly zero
        mask = torch.ones(rep.shape[0], 1, device=rep.device, dtype=rep.dtype)
        mask[self.pad_id] = 0.0
        return rep * mask

    def embed_items(self, item_seq: torch.Tensor) -> torch.Tensor:
        return self.item_emb(item_seq) + self.text_proj(self.text_feat[item_seq])

    def output_item_weights(self) -> torch.Tensor:
        return self.item_emb.weight[:-1] + self.text_proj(self.text_feat[:-1])
