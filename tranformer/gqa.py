"""Grouped Query Attention"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange


class GQA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_groups: int) -> None:
        super().__init__()
        d_head = d_model // n_heads
        self.d_head = torch.tensor(d_head)
        self.d_model = torch.tensor(d_model)
        self.n_heads = torch.tensor(n_heads)
        self.n_groups = torch.tensor(n_groups)

        self.group_size = n_heads // n_groups

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_head * n_groups)
        self.w_v = nn.Linear(d_model, d_head * n_groups)
        self.w_out = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        b, s, d = x.shape
        q = self.w_q(x)  # b,seq,model_dim
        k = self.w_k(x)  # b,seq, groups*d_head
        v = self.w_v(x)

        q = rearrange(q, "b s (n_h h_dim) -> b n_h s h_dim", n_h=self.n_heads)
        k: Tensor = rearrange(k, "b s (g h_dim) -> b g s h_dim", g=self.n_groups)
        v = rearrange(v, "b s (g h_dim) -> b g s h_dim", g=self.n_groups)

        # interleave k and v
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)  # b, heads, s, h_dim

        scores = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(self.d_head)
        att = torch.nn.functional.softmax(scores, dim=-1)
        x = torch.matmul(att, v)
        x = rearrange(x, "b heads s h_dim -> b s (heads h_dim)")
        x = self.w_out(x)
        return x


if __name__ == "__main__":
    batch = 5
    seq = 16
    dim = 128
    n_heads = 16
    n_groups = 8
    attention = GQA(d_model=dim, n_heads=n_heads, n_groups=n_groups)
    x = torch.ones(batch, seq, dim)

    x = attention(x)
