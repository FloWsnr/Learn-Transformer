from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from einops import rearrange


class KVCache:
    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.k_cache = torch.empty(
            batch_size, max_seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        self.v_cache = torch.empty(
            batch_size, max_seq_len, num_heads, head_dim, device=device, dtype=dtype
        )

        self.current_seq_len = 0

    def update(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the key and value caches with new key and value tensors.

        Parameter
        ---------

        k: torch.Tensor
            key to cache (B, NH, SEQ, HD)

        v: torch.Tensor
            value to cache (B, NH, SEQ, HD)
        """

        seq_len = k.shape[1]
        new_len = self.current_seq_len + seq_len
        self.k_cache[:, :, self.current_seq_len : new_len, :]
        self.k_cache[:, :, self.current_seq_len : new_len, :]

        self.current_seq_len = new_len

        return k, v


class MHA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int) -> None:
        super().__init__()

        d_head = d_model // n_heads
        self.d_head = torch.tensor(d_head)
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)

        self.kv_cache: Optional[KVCache] = None

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if self.kv_cache is None:
            b, seq, model_dim = x.shape
            self.kv_cache = KVCache(
                batch_size=b,
                max_seq_len=self.max_seq_len,
                num_heads=self.n_heads,
                head_dim=int(self.d_head),
                device=x.device,
                dtype=x.dtype,
            )

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = rearrange(
            q, "b seq (heads d_head) -> b heads seq d_head", heads=self.n_heads
        )

        k = rearrange(
            k, "b seq (heads d_head) -> b heads seq d_head", heads=self.n_heads
        )

        v = rearrange(
            v, "b seq (heads d_head) -> b heads seq d_head", heads=self.n_heads
        )

        k, v = self.kv_cache.update(k, v)

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / torch.sqrt(self.d_head)

        if mask is not None:
            scores = torch.masked_fill(scores, mask=mask, value=-torch.inf)

        att = nn.functional.softmax(scores, dim=-1)

        x = torch.matmul(att, v)
        x = rearrange(
            x, "b heads seq d_head -> b seq (heads d_head)", heads=self.n_heads
        )

        x = self.w_out(x)

        return x


if __name__ == "__main__":
    batch = 5
    seq = 16
    dim = 16
    n_heads = 2
    attention = MHA(d_model=dim, n_heads=n_heads)
    x = torch.ones(batch, seq, dim)

    x = attention(x)
    x = attention(x)
