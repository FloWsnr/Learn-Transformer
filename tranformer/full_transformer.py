import torch
from torch import Tensor, nn
import math


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()

        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: Tensor) -> Tensor:

        var, mu = torch.var_mean(
            x, dim=-1, correction=0, keepdim=True
        )  # no bias, since we want to have the full variane of THIS batch / features
        x = (x - mu) / torch.sqrt(var + self.eps)

        return x * self.gamma + self.beta


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8) -> None:
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:

        norm = torch.pow(x, 2).mean(dim=-1, keepdim=True)
        x = x / torch.sqrt(norm + self.eps)
        return x * self.gamma


class MLP(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.W_in = nn.Linear(d_model, hidden_dim)
        self.W_out = nn.Linear(hidden_dim, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act(self.W_in(x))
        x = self.dropout(x)
        return self.W_out(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.W_vg = nn.Linear(d_model, 2 * hidden_dim)
        self.W_out = nn.Linear(hidden_dim, d_model)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        vg: Tensor = self.W_vg(x)
        v, g = vg.chunk(2, dim=-1)
        g = self.act(g)

        x = self.dropout(v * g)
        x = self.W_out(x)
        return x


class MHA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.W_in = nn.Linear(d_model, 3 * d_model)
        self.dropout = nn.Dropout(dropout)
        self.W_out = nn.Linear(d_model, d_model)

        self.d_heads = d_model // n_heads
        self.n_heads = n_heads
        self.scale = self.d_heads**-0.5

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        b, s, d = x.shape

        qkv: Tensor = self.W_in(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape into heads
        q = q.view(b, s, self.n_heads, self.d_heads).transpose(1, 2)
        k = k.view(b, s, self.n_heads, self.d_heads).transpose(1, 2)
        v = v.view(b, s, self.n_heads, self.d_heads).transpose(1, 2)

        # attention
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            scores = torch.masked_fill(scores, mask, value=-torch.inf)
        scores = torch.nn.functional.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        x = torch.matmul(scores, v)
        # reshape
        x = x.transpose(1, 2).reshape(b, s, -1)
        x = self.W_out(x)

        return x


class AbsEmbeddings(nn.Module):
    def __init__(self, d_model: int, max_seq: int) -> None:
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(max_seq, d_model))

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.shape[-2]
        return x + self.pe[:seq_len, :]


class SineEmbeddings(nn.Module):
    def __init__(self, d_model: int, max_seq: int) -> None:
        super().__init__()

        pe = torch.zeros(max_seq, d_model)

        pos = torch.arange(0, max_seq).unsqueeze(-1)
        dims = torch.arange(0, d_model, 2)
        freq = torch.exp(-math.log(10000) * dims / d_model)

        pe[:, ::2] = torch.sin(pos * freq)
        pe[:, 1::2] = torch.cos(pos * freq)
        self.register_buffer("pe", pe)
        self.pe: Tensor

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.shape[-2]
        return x + self.pe[:seq_len, :]


class RoPE(nn.Module):
    def __init__(self, d_model: int, max_seq: int) -> None:
        super().__init__()

        freq = torch.arange(0, d_model, 2) / d_model
        freq = torch.exp(freq * -math.log(10000))
        pos = torch.arange(0, max_seq).unsqueeze(-1)
        angles = pos * freq
        self.register_buffer("cos", angles.cos())
        self.register_buffer("sin", angles.sin())
        self.cos: Tensor
        self.sin: Tensor

    def forward(self, x):
        b, seq, d = x.shape
        sin = self.sin[:seq]
        cos = self.cos[:seq]

        x1 = x[::2]
        x2 = x[1::2]

        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        return torch.stack((out1, out2), dim=-1).flatten(-2)


class Router(nn.Module):
    def __init__(self, d_in: int, n_experts: int, top_k: int) -> None:
        super().__init__()

        self.routing = nn.Linear(d_in, n_experts)
        self.top_k = top_k

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:

        logits = self.routing(x)
        vals, choices = torch.topk(logits, self.top_k, dim=-1)
        weights = nn.functional.softmax(vals, dim=-1)

        return choices, weights


class MOE(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, n_experts: int, top_k: int) -> None:
        super().__init__()

        self.experts = nn.ModuleList(
            [SwiGLU(d_model=d_in, hidden_dim=d_hidden) for _ in range(n_experts)]
        )
        self.n_experts = n_experts
        self.shared_expert = SwiGLU(d_model=d_in, hidden_dim=d_hidden)

        self.router = Router(d_in, n_experts, top_k=top_k)
        self.top_k = top_k

    def forward(self, x: Tensor) -> Tensor:

        # choose experts by router
        choices, weights = self.router(x)

        output = torch.zeros_like(x)

        # iterate over experts
        for k in range(self.top_k):
            expert_idx = choices[..., k]  # B, S
            weight = weights[..., k]  # B, S

            for e in range(self.n_experts):
                mask = expert_idx == e  # B,S
                inp = x[mask, :]  # N=number of times, this expert is chosen, DIM
                w = weight[mask]  # N,
                out = self.experts[e](inp)

                output[mask, :] += w.unsqueeze(-1) * out

        shared = self.shared_expert(x)

        return shared + output


class TokenEmbedding(nn.Module):
    def __init__(self, n_tokens: int, d_model: int) -> None:
        super().__init__()

        self.emb = nn.Embedding(
            n_tokens, embedding_dim=d_model
        )  # lookup table of integer to vectors

    def forward(self, toks: Tensor) -> Tensor:
        return self.emb(toks)


class OutputLayer(nn.Module):
    def __init__(self, n_tokens: int, d_model: int) -> None:
        super().__init__()

        self.de_emb = nn.Linear(d_model, n_tokens)

    def forward(self, x: Tensor) -> Tensor:

        x = self.de_emb(x)
        return x
