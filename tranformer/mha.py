from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange


class MHA(nn.Module):
    def __init__(self, model_dim: int, n_heads: int) -> None:
        super().__init__()
        self.w_q = nn.Linear(model_dim, model_dim)
        self.w_k = nn.Linear(model_dim, model_dim)
        self.w_v = nn.Linear(model_dim, model_dim)

        self.w_out = nn.Linear(model_dim, model_dim)

        self.model_dim = torch.tensor(model_dim)
        self.n_heads = n_heads
        self.head_dim = torch.tensor(model_dim // n_heads)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch, seq, dim = x.shape

        q: torch.Tensor = self.w_q(x)
        k: torch.Tensor = self.w_k(x)
        v: torch.Tensor = self.w_v(x)

        q = rearrange(
            q,
            "B seq (heads h_dim) -> B heads seq h_dim",
            heads=self.n_heads,
            h_dim=self.head_dim,
        )
        k = rearrange(
            k,
            "B seq (heads h_dim) -> B heads seq h_dim",
            heads=self.n_heads,
            h_dim=self.head_dim,
        )
        v = rearrange(
            v,
            "B seq (heads h_dim) -> B heads seq h_dim",
            heads=self.n_heads,
            h_dim=self.head_dim,
        )

        att = torch.matmul(q, k.transpose(-2, -1))  # B heads seq seq
        att = att / torch.sqrt(self.head_dim)
        if mask is not None:
            att.masked_fill_(mask, -torch.inf)
        att = nn.functional.softmax(att, dim=-1)

        x = torch.matmul(att, v)  # B heads seq head_dim

        # back to full model_dim
        x = rearrange(x, "B heads seq h_dim -> B seq (heads h_dim)")
        x = self.w_out(x)

        return x


if __name__ == "__main__":
    batch = 5
    seq = 16
    dim = 16
    n_heads = 2
    attention = MHA(model_dim=dim, n_heads=n_heads)
    x = torch.ones(batch, seq, dim)

    x = attention(x)
