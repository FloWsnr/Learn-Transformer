"""Mixture of experts implementation"""

import torch
from torch import nn, Tensor


class MOE(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        n_experts: int,
        top_k: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # create experts
        self.n_experts = n_experts
        self.experts = nn.ModuleList(
            [
                SwiGLU(d_in=d_in, d_hidden=d_hidden, dropout=dropout)
                for _ in range(n_experts)
            ]
        )
        self.shared_expert = SwiGLU(d_in=d_in, d_hidden=d_hidden, dropout=dropout)
        self.router = Router(d_in=d_in, n_experts=n_experts, top_k=top_k)
        self.top_k = top_k

    def forward(self, x: Tensor) -> Tensor:
        choices, weights = self.router(x)  # b, s, topk
        # choices_f = choices.reshape(-1, self.top_k)  # (b s) topk
        # weights_f = weights.reshape(-1, self.top_k)

        output = torch.zeros_like(x)

        for i in range(self.top_k):
            idx_expert = choices[..., i]  # B, S
            weight = weights[..., i]

            for e in range(self.n_experts):
                mask = idx_expert == e  # B, S bool
                if mask.any():
                    inp = x[mask]
                    out = self.experts[e](inp)
                    w = weight[mask]
                    output[mask] += w.unsqueeze(-1) * out

        shared_out = self.shared_expert(x)

        return shared_out + output


class SwiGLU(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.w_in = nn.Linear(d_in, 2 * d_hidden)
        self.w_out = nn.Linear(d_hidden, d_in)

    def forward(self, x: Tensor) -> Tensor:
        v, g = self.w_in(x).chunk(2, dim=-1)
        g = torch.nn.functional.silu(g)

        x = v * g
        x = self.dropout(x)
        x = self.w_out(x)

        return x


class Router(nn.Module):
    def __init__(self, d_in: int, n_experts: int, top_k: int) -> None:
        super().__init__()
        self.router = nn.Linear(d_in, n_experts)
        self.softmax = nn.Softmax(dim=-1)
        self.top_k = top_k

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Receives embedded tokens, returns indices of chosen experts and their weight"""
        choices = self.router(x)  # B, Seq, Experts
        top_values, top_choices = torch.topk(choices, k=self.top_k, dim=-1)
        weights = self.softmax(top_values)

        return top_choices, weights


if __name__ == "__main__":
    b = 16
    seq = 32
    dim = 8

    x = torch.rand(b, seq, dim)

    moe = MOE(d_in=dim, d_hidden=2 * dim, n_experts=4, top_k=2, dropout=0.1)
    x = moe(x)
