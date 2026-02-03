import math

import torch
from torch import Tensor, nn


class SinuEmbeddings(nn.Module):
    def __init__(self, n: int, d_model: int) -> None:
        super().__init__()
        self.pe = torch.zeros(n, d_model)

        pos = torch.arange(n).unsqueeze(-1)

        steps = torch.arange(0, d_model, 2) / d_model
        div = torch.exp(-math.log(10000) * steps)

        self.pe[:, 0::2] = torch.sin(pos * div)
        self.pe[:, 1::2] = torch.cos(pos * div)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.shape[-2]
        x = x + self.pe[:, :seq_len, :]

        return x


class RoPE(nn.Module):
    def __init__(self, d_model: int, max_l: int, base: int = 10000) -> None:
        super().__init__()

        steps = torch.arange(0, d_model, 2) / d_model
        pos = torch.arange(0, max_l)
        div = torch.exp(-math.log(base) * steps)

        angles = pos.unsqueeze(-1) * div.unsqueeze(0)
        self.register_buffer("cos", angles.cos())
        self.register_buffer("sin", angles.sin())
        self.cos: Tensor
        self.sin: Tensor

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.shape[-2]
        cos = self.cos[:seq_len, :]
        sin = self.sin[:seq_len, :]

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        # rotation
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        return torch.stack((out1, out2), dim=-1).flatten(-2)


if __name__ == "__main__":
    tokens = torch.rand(16, 32, 8)
    emb = SinuEmbeddings(n=1000, d_model=8)

    emb_toks = emb(tokens)
