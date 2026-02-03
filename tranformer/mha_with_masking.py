"""Multihead Attention with causal masking and pad-masking"""

from typing import Optional

import torch
from torch import Tensor, nn


class MHA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()

        assert d_model % n_heads == 0, "Error"
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.scale = self.d_head**-0.5

        self.W_in = nn.Linear(d_model, 3 * d_model)
        self.W_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        b, seq, d_in = x.shape

        qkv: Tensor = self.W_in(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # rearange into heads
        q = q.reshape(b, seq, self.n_heads, self.d_head).transpose(1, 2)
        k = k.reshape(b, seq, self.n_heads, self.d_head).transpose(1, 2)
        v = v.reshape(b, seq, self.n_heads, self.d_head).transpose(1, 2)

        # attention
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            scores = torch.masked_fill(scores, mask=mask, value=-torch.inf)

        scores = torch.nn.functional.softmax(scores, dim=-1)
        # dropout
        scores = self.dropout(scores)

        att = torch.matmul(scores, v)

        # reshape
        att = att.transpose(1, 2).reshape(b, seq, d_in)

        x = self.W_out(att)
        return x


if __name__ == "__main__":
    batch = 5
    seq = 16
    dim = 4
    n_heads = 2
    toks = torch.ones(batch, seq)
    attention = MHA(d_model=dim, n_heads=n_heads)
    x = torch.ones(batch, seq, dim)

    pad_mask = toks == 0
    pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)  # B, 1, 1, Seq
    causal_mask = ~torch.tril(torch.ones(seq, seq, dtype=torch.bool))
    mask = torch.logical_or(causal_mask, pad_mask)

    x = attention(x, mask)
    print(x.shape)
