"""BERT4Rec (Sun et al., CIKM 2019): bidirectional Transformer + Cloze (masked item) training.

Token ids: items 0..n-1, pad = n, [MASK] = n+1. Training masks each non-pad
position with prob ``mask_ratio`` (at least one per sequence) and applies a
full-catalogue CE at masked positions. At evaluation the history (last
max_len-1 items) is followed by a [MASK] token and the logits at that
position rank the whole catalogue.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sasrec import SASRecBlock


class BERT4Rec(nn.Module):
    def __init__(self, n_items: int, hidden_size: int = 64, n_layers: int = 2, n_heads: int = 2,
                 inner_size: int = 256, max_seq_length: int = 50, hidden_dropout_prob: float = 0.2,
                 attn_dropout_prob: float = 0.2, mask_ratio: float = 0.2, initializer_range: float = 0.02):
        super().__init__()
        self.n_items = n_items
        self.pad_id = n_items
        self.mask_id = n_items + 1
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.mask_ratio = mask_ratio
        self.initializer_range = initializer_range
        self.item_emb = nn.Embedding(n_items + 2, hidden_size, padding_idx=self.pad_id)
        self.pos_emb = nn.Embedding(max_seq_length, hidden_size)
        self.emb_dropout = nn.Dropout(hidden_dropout_prob)
        self.blocks = nn.ModuleList([
            SASRecBlock(hidden_size, n_heads, inner_size, attn_dropout_prob, hidden_dropout_prob)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_size)
        # output transform (as in BERT's MLM head) + per-item output bias; output matrix tied to input emb
        self.out_dense = nn.Linear(hidden_size, hidden_size)
        self.out_norm = nn.LayerNorm(hidden_size)
        self.out_bias = nn.Parameter(torch.zeros(n_items))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            m.weight.data.normal_(0.0, self.initializer_range)
        if isinstance(m, nn.Linear) and m.bias is not None:
            m.bias.data.zero_()
        if isinstance(m, nn.Embedding) and m.padding_idx is not None:
            m.weight.data[m.padding_idx].zero_()
        if isinstance(m, nn.LayerNorm):
            m.bias.data.zero_()
            m.weight.data.fill_(1.0)

    def forward(self, item_seq: torch.Tensor) -> torch.Tensor:
        B, L = item_seq.shape
        pos = torch.arange(L, device=item_seq.device).unsqueeze(0).expand(B, -1)
        x = self.emb_dropout(self.item_emb(item_seq) + self.pos_emb(pos))
        valid = item_seq.ne(self.pad_id)
        block = (~valid).unsqueeze(1).expand(B, L, L)  # block pad keys only (bidirectional)
        bias = torch.zeros(B, 1, L, L, device=item_seq.device).masked_fill(block.unsqueeze(1), -1e4)
        x = x * valid.unsqueeze(-1).to(x.dtype)
        for blk in self.blocks:
            x = blk(x, attn_bias=bias, pad_mask=valid)
        x = self.final_norm(x)
        return x * valid.unsqueeze(-1).to(x.dtype)

    def output_item_weights(self) -> torch.Tensor:
        return self.item_emb.weight[: self.n_items]

    def head(self, h: torch.Tensor) -> torch.Tensor:
        h = self.out_norm(F.gelu(self.out_dense(h)))
        return h @ self.output_item_weights().t() + self.out_bias

    def mask_batch(self, seq: torch.Tensor, generator: torch.Generator | None = None):
        """Return (masked_input, targets with -100 at unmasked positions)."""
        valid = seq.ne(self.pad_id)
        prob = torch.rand(seq.shape, generator=generator, device=seq.device)
        m = (prob < self.mask_ratio) & valid
        # guarantee >=1 masked position per non-empty row: mask the last item if none chosen
        none = ~m.any(1) & valid.any(1)
        if none.any():
            m[none, -1] = True
        inp = seq.masked_fill(m, self.mask_id)
        tgt = seq.masked_fill(~m, -100)
        return inp, tgt

    def append_mask(self, item_seq: torch.Tensor) -> torch.Tensor:
        """Left-padded history [B, L] -> last L-1 items followed by [MASK]."""
        B = item_seq.shape[0]
        m = torch.full((B, 1), self.mask_id, dtype=item_seq.dtype, device=item_seq.device)
        return torch.cat([item_seq[:, 1:], m], dim=1)

    def predict_logits(self, item_seq: torch.Tensor) -> torch.Tensor:
        h = self.forward(self.append_mask(item_seq))[:, -1, :]
        return self.head(h)

    def score(self, users, item_seq):
        return self.predict_logits(item_seq)

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)
