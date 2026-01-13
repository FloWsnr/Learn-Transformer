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

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape

        q = self.w_q(x)
        k: torch.Tensor = self.w_k(x)
        v = self.w_v(x)

        att = torch.matmul(q, k.view(0, 2, 1))
        att = att / torch.sqrt(self.model_dim)
        att = nn.functional.softmax(att, dim=-1)

        x = torch.matmul(att, v)
        x = self.w_out(x)
        return x


if __name__ == "__main__":
    batch = 5
    seq = 16
    dim = 16
    attention = Attention(model_dim=dim)
    x = torch.ones(batch, seq, dim)

    x = attention(x)