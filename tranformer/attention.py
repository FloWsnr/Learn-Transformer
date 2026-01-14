from typing import Optional

import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.w_q = nn.Linear(model_dim, model_dim)
        self.w_k = nn.Linear(model_dim, model_dim)
        self.w_v = nn.Linear(model_dim, model_dim)

        self.w_out = nn.Linear(model_dim, model_dim)

        self.model_dim = torch.tensor(model_dim)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch, seq, dim = x.shape

        q: torch.Tensor = self.w_q(x)
        k: torch.Tensor = self.w_k(x)
        v: torch.Tensor = self.w_v(x)

        att = torch.matmul(q, k.transpose(-2, -1))
        att = att / torch.sqrt(self.model_dim)
        if mask is not None:
            att.masked_fill_(mask, -torch.inf)
        att = nn.functional.softmax(att, dim=-1)

        x = torch.matmul(att, v)
        x = self.w_out(x)
        return x
