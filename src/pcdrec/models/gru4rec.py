"""GRU4Rec (Hidasi et al., ICLR 2016) with full-catalogue softmax CE (RecBole-style).

Embedding (d) -> dropout -> GRU (hidden) -> Linear(hidden -> d); scores are
dot products with the (tied) item embedding. Inputs are left-padded like
SASRec; internally the sequence is packed so padding never enters the GRU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GRU4Rec(nn.Module):
    def __init__(self, n_items: int, hidden_size: int = 64, gru_hidden: int = 64, n_layers: int = 1,
                 dropout: float = 0.3, max_seq_length: int = 50, initializer_range: float = 0.02):
        super().__init__()
        self.n_items = n_items
        self.pad_id = n_items
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.item_emb = nn.Embedding(n_items + 1, hidden_size, padding_idx=self.pad_id)
        self.emb_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, gru_hidden, num_layers=n_layers, batch_first=True, bias=False)
        self.dense = nn.Linear(gru_hidden, hidden_size)
        nn.init.xavier_normal_(self.item_emb.weight)
        self.item_emb.weight.data[self.pad_id].zero_()
        for name, p in self.gru.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(p)
        nn.init.xavier_normal_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)

    def forward(self, item_seq: torch.Tensor) -> torch.Tensor:
        """item_seq [B, L] left-padded -> hidden [B, L, d] aligned with input positions (pad rows = 0)."""
        B, L = item_seq.shape
        valid = item_seq.ne(self.pad_id)
        lengths = valid.sum(1).clamp(min=1)
        offset = (L - lengths).unsqueeze(1)  # number of leading pads
        ar = torch.arange(L, device=item_seq.device).unsqueeze(0)
        right_idx = (ar + offset).clamp(max=L - 1)
        right = item_seq.gather(1, right_idx)
        right = right.masked_fill(ar >= lengths.unsqueeze(1), self.pad_id)
        x = self.emb_dropout(self.item_emb(right))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=L)
        out = self.dense(out)  # [B, L, d], right-aligned
        left_idx = (ar - offset).clamp(min=0)
        left = out.gather(1, left_idx.unsqueeze(-1).expand(-1, -1, out.size(-1)))
        return left * valid.unsqueeze(-1).to(left.dtype)

    def output_item_weights(self) -> torch.Tensor:
        return self.item_emb.weight[:-1]

    def predict_logits(self, item_seq: torch.Tensor) -> torch.Tensor:
        h = self.forward(item_seq)[:, -1, :]  # left-padded: last position is the most recent item
        return h @ self.output_item_weights().t()

    def score(self, users, item_seq):
        return self.predict_logits(item_seq)

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)
