import torch


class TransformerLayer(torch.nn.Module):
    def __init__(self):
        super(TransformerLayer, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(512, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 512),
        )

        self.q = torch.nn.Linear(512, 100)
        self.k = torch.nn.Linear(512, 100)
        self.v = torch.nn.Linear(512, 100)

    def forward(self, x):
        # Multi-head attention
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # calc attention
        attention = torch.matmul(q, k.T)
        attention = attention / torch.sqrt(torch.tensor(100.0))
        attention = torch.softmax(attention, dim=-1)
        attention = torch.matmul(attention, v)

        # Normalization
        x = x + attention  # residual
        x = torch.nn.LayerNorm(x)

        # Feed forward
        x_new = self.mlp(x)

        # Normalization
        x = x_new + x  # residual
        x = torch.nn.LayerNorm(x)

        return x


class AttentionHead(torch.nn.Module):
    def __init__(self):
        super(AttentionHead, self).__init__()

    def forward(self, x):
        return x
