"""Causal self-attention SASRec (Kang & McAuley, ICDM 2018) with full-catalog CE head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_size: int, inner_size: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, inner_size)
        self.fc2 = nn.Linear(inner_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class SASRecBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        inner_size: int,
        attn_dropout: float,
        hidden_dropout: float,
    ):
        super().__init__()
        assert hidden_size % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.hidden_size = hidden_size

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.ffn = PointWiseFeedForward(hidden_size, inner_size, hidden_dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, H]
        attn_bias: [B, 1, L, L] additive mask (0 = keep, large negative = block)
        pad_mask: [B, L] True for valid (non-pad) tokens
        """
        B, L, H = x.shape
        residual = x
        x = self.norm1(x)
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        # scores: [B, heads, L, L]
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = attn + attn_bias
        attn = torch.softmax(attn, dim=-1)
        # padded queries can be all-blocked -> NaN; zero them
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)  # [B, heads, L, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, L, H)
        out = self.out_proj(out)
        x = residual + self.dropout(out)
        x = x * pad_mask.unsqueeze(-1).to(x.dtype)

        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ffn(x))
        x = x * pad_mask.unsqueeze(-1).to(x.dtype)
        return x


class SASRec(nn.Module):
    def __init__(
        self,
        n_items: int,
        hidden_size: int = 64,
        n_layers: int = 2,
        n_heads: int = 2,
        inner_size: int = 256,
        max_seq_length: int = 50,
        hidden_dropout_prob: float = 0.2,
        attn_dropout_prob: float = 0.2,
        initializer_range: float = 0.02,
    ):
        super().__init__()
        self.n_items = n_items
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.inner_size = inner_size

        self.pad_id = n_items
        self.item_emb = nn.Embedding(n_items + 1, hidden_size, padding_idx=self.pad_id)
        self.pos_emb = nn.Embedding(max_seq_length, hidden_size)
        self.emb_dropout = nn.Dropout(hidden_dropout_prob)

        self.blocks = nn.ModuleList(
            [
                SASRecBlock(
                    hidden_size=hidden_size,
                    n_heads=n_heads,
                    inner_size=inner_size,
                    attn_dropout=attn_dropout_prob,
                    hidden_dropout=hidden_dropout_prob,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.initializer_range = initializer_range
        self.apply(self._init_weights)

        # causal: True where j > i (block future)
        causal = torch.triu(torch.ones(max_seq_length, max_seq_length, dtype=torch.bool), diagonal=1)
        self.register_buffer("_causal_bool", causal, persistent=False)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        if isinstance(module, nn.Embedding) and module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _attn_bias(self, item_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return additive attn bias [B,1,L,L] and pad_mask [B,L] (True=valid)."""
        B, L = item_seq.shape
        pad_mask = item_seq.ne(self.pad_id)  # True = valid
        # block future
        causal = self._causal_bool[:L, :L]  # [L,L]
        # block pad keys: [B, L]
        key_pad = ~pad_mask
        # combine: [B, L, L] True = block
        block = causal.unsqueeze(0) | key_pad.unsqueeze(1)
        # also block pad queries from affecting anything meaningfully (optional)
        bias = torch.zeros(B, 1, L, L, device=item_seq.device, dtype=torch.float32)
        bias = bias.masked_fill(block.unsqueeze(1), -1e4)
        return bias, pad_mask

    def forward(self, item_seq: torch.Tensor) -> torch.Tensor:
        """
        item_seq: [B, L] long, left-padded with pad_id; recent items at the end.
        returns: [B, L, H]
        """
        B, L = item_seq.shape
        device = item_seq.device
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        x = self.item_emb(item_seq) + self.pos_emb(positions)
        x = self.emb_dropout(x)
        attn_bias, pad_mask = self._attn_bias(item_seq)
        x = x * pad_mask.unsqueeze(-1).to(x.dtype)
        for block in self.blocks:
            x = block(x, attn_bias=attn_bias, pad_mask=pad_mask)
        x = self.final_norm(x)
        x = x * pad_mask.unsqueeze(-1).to(x.dtype)
        return x

    def user_representation(self, item_seq: torch.Tensor) -> torch.Tensor:
        """Last non-pad position hidden state -> [B, H].

        Sequences are left-padded, so the last valid token is at the rightmost
        non-pad index (not at length-1 from the left).
        """
        h = self.forward(item_seq)  # [B, L, H]
        valid = item_seq.ne(self.pad_id)  # [B, L]
        # index of last True along L; if all pad, fall back to 0
        # flip + argmax finds first True from the right
        last = valid.size(1) - 1 - valid.flip(dims=[1]).long().argmax(dim=1)
        last = last.clamp(min=0)
        idx = last.view(-1, 1, 1).expand(-1, 1, h.size(-1))
        return h.gather(1, idx).squeeze(1)

    def predict_logits(self, item_seq: torch.Tensor) -> torch.Tensor:
        """Full-catalog logits [B, n_items]."""
        u = self.user_representation(item_seq)
        w = self.item_emb.weight[:-1]
        return u @ w.t()

    def train_step_logits(self, seq_in: torch.Tensor) -> torch.Tensor:
        """logits [B, L, n_items] for each position."""
        h = self.forward(seq_in)
        w = self.item_emb.weight[:-1]
        return h @ w.t()

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters() if not trainable_only else (p for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in params)
