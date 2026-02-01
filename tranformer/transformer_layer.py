import torch
from torch import Tensor, nn


class SwiGLU(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.W_in = nn.Linear(d_in, 2 * d_hidden)
        self.W_out = nn.Linear(d_hidden, d_in)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        vg: Tensor = self.W_in(x)
        v, g = vg.chunk(2, dim=-1)

        g = nn.functional.silu(g)
        x = v * g
        # apply dropout
        x = self.dropout(x)
        # back to d_in
        x = self.W_out(x)
        return x


class MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_in, d_hidden)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class MHA(nn.Module):
    def __init__(self, d_in: int, n_head: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_in % n_head == 0, "Model dim not divisible by number of heads"
        self.d_head = d_in // n_head
        self.d_in = d_in
        self.n_heads = n_head

        self.scale = self.d_head**-0.5
        self.W_in = nn.Linear(d_in, 3 * d_in)
        self.W_out = nn.Linear(d_in, d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        b, seq, d_m = x.shape
        qkv: Tensor = self.W_in(x)

        # reshape for heads
        qkv = qkv.view(b, seq, self.n_heads, self.d_head * 3)
        qkv = qkv.transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=-1)

        # attention:
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=self.scale)
        # scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # scores = torch.softmax(scores, dim=-1)
        # scores = self.dropout(scores)

        # x = torch.matmul(scores, v)

        # reshape back
        x = x.transpose(1, 2).contiguous()
        x = x.view(b, seq, -1)

        x = self.W_out(x)

        return x


if __name__ == "__main__":
    batch = 5
    seq = 16
    dim = 16
    n_heads = 2
    attention = MHA(d_in=dim, n_head=n_heads)
    x = torch.ones(batch, seq, dim)

    x = attention(x)
