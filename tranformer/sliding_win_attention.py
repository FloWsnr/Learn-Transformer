import torch
from torch import Tensor
from einops import rearrange


class SWMQA(torch.nn.Module):
    def __init__(
        self, d_model: int, n_kv_heads: int, n_heads: int, max_seq: int, window: int
    ) -> None:
        super().__init__()

        d_head = d_model // n_heads
        self.scale = d_head**-0.5
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.group_size = n_heads // n_kv_heads

        # one full q and two times reduced num of heads for k and v
        self.splits = [d_model, d_head * n_kv_heads, d_head * n_kv_heads]
        self.w_in = torch.nn.Linear(d_model, sum(self.splits))
        self.w_out = torch.nn.Linear(d_model, d_model)

        # causal mask
        mask = torch.ones(max_seq, max_seq, dtype=torch.bool)
        mask = torch.tril(mask)
        row = torch.arange(max_seq).unsqueeze(1)
        col = torch.arange(max_seq).unsqueeze(0)
        distance = row - col

        mask = mask & (distance < window)
        self.register_buffer("mask", mask)
        self.mask: Tensor

    def forward(self, x: Tensor) -> Tensor:
        b, s, n = x.shape

        qkv: Tensor = self.w_in(x)
        q, k, v = qkv.split_with_sizes(self.splits, dim=-1)
        # rearange
        q = rearrange(q, "b s (d_h n_h) -> b n_h s d_h", n_h=self.n_heads)
        k = rearrange(k, "b s (d_h n_h) -> b n_h s d_h", n_h=self.n_kv_heads)
        v = rearrange(v, "b s (d_h n_h) -> b n_h s d_h", n_h=self.n_kv_heads)

        k = torch.repeat_interleave(k, repeats=self.group_size, dim=1)
        v = torch.repeat_interleave(v, repeats=self.group_size, dim=1)
        # attention
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # masks
        mask = self.mask[:s, :s]
        scores = torch.masked_fill(scores, mask == 0, value=-torch.inf)
        scores = torch.softmax(scores, dim=-1)

        x = scores @ v
        x = rearrange(x, "b n_h s d_h -> b s (n_h d_h)")
        x = self.w_out(x)

        return x


if __name__ == "__main__":
    batch = 5
    seq = 16
    dim = 128
    n_heads = 16
    attention = SWMQA(d_model=dim, n_kv_heads=4, n_heads=n_heads, max_seq=32, window=8)
    x = torch.ones(batch, seq, dim)

    x = attention(x)
