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
        self.k_cache = torch.zeros(
            batch_size, num_heads, max_seq_len, head_dim, device=device, dtype=dtype
        )
        self.v_cache = torch.zeros(
            batch_size, num_heads, max_seq_len, head_dim, device=device, dtype=dtype
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

        seq_len = k.shape[2]
        new_len = self.current_seq_len + seq_len
        self.k_cache[:, :, self.current_seq_len : new_len, :] = k
        self.v_cache[:, :, self.current_seq_len : new_len, :] = v

        self.current_seq_len = new_len

        return (self.k_cache[:, :, :new_len, :], self.v_cache[:, :, :new_len, :])

    def reset(self) -> None:
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.current_seq_len = 0


class MHA(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()

        d_head = d_model // n_heads
        self.d_head = d_head
        self.n_heads = n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: Tensor,
        kv_cache: Optional[KVCache] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
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

        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / torch.sqrt(torch.tensor(self.d_head, dtype=scores.dtype))

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
    max_seq = 16
    prompt_l = 10

    dim = 8
    n_heads = 2
    text = torch.ones(batch, prompt_l, dim)

    attention = MHA(d_model=dim, n_heads=n_heads)
    kv_cache = KVCache(
        batch_size=batch,
        max_seq_len=max_seq,
        num_heads=n_heads,
        head_dim=dim // n_heads,
        device=text.device,
        dtype=text.dtype,
    )

    # prefil
    mask = torch.ones(batch, 1, prompt_l, prompt_l, dtype=torch.bool)
    mask = torch.triu(
        mask
    )  # we need upper, since True elements are masked_filled with -inf
    out = attention(text, kv_cache, mask=mask)

    for _ in range(6):
        new_token = torch.ones(batch, 1, dim)
        out = attention(new_token, kv_cache=kv_cache)

    kv_cache.reset()
