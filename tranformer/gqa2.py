"""Second version of grouped query attention"""

from typing import Optional

import torch
from torch import Tensor, nn


class GQA(nn.Module):
    """Grouped Query Attention

    GQA uses a single key & value for multiple query heads, reducing computation and KV cache requirements
    """

    def __init__(
        self, d_model: int, n_heads: int, n_groups: int, dropout: float = 0.0
    ) -> None:
        super().__init__()

        assert d_model % n_heads == 0, ""
        assert n_heads % n_groups == 0, ""

        d_head = d_model // n_heads
        self.d_head = d_head
        self.scale = d_head**-0.5
        self.group_size = n_heads // n_groups

        self.dropout = nn.Dropout(dropout)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, n_groups * d_head)
        self.W_v = nn.Linear(d_model, n_groups * d_head)
        self.W_out = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        b, seq, dm = x.shape

        # QKV
        q = self.W_q(x)  # b, seq, d_model
        k = self.W_k(x)  # b, seq, n_groups * d_heads
        v = self.W_v(x)  # b, seq, n_groups * d_heads

        # reshape
        q = q.reshape(b, seq, -1, self.d_head).transpose(1, 2)
        k = k.reshape(b, seq, -1, self.d_head).transpose(1, 2)
        v = v.reshape(b, seq, -1, self.d_head).transpose(1, 2)

        # interleave k and v
        k = torch.repeat_interleave(k, self.group_size, dim=1)
        v = torch.repeat_interleave(v, self.group_size, dim=1)

        # attention
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            scores = torch.masked_fill(scores, mask=mask, value=-torch.inf)

        scores = torch.nn.functional.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        att = torch.matmul(scores, v)
        # reshape
        att = att.transpose(1, 2).reshape(b, seq, -1)

        x = self.W_out(att)

        return x


if __name__ == "__main__":
    x = torch.rand(16, 32, 64)
    gqa = GQA(d_model=64, n_heads=8, n_groups=2)

    x = gqa(x)
