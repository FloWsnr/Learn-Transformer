import torch

from learn_former.model.submodels import (
    MultiHeadAttention,
    FeedForward,
)


class TransformerLayer(torch.nn.Module):
    def __init__(self):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention()
        self.feed_forward = FeedForward()

        self.layer_norm1 = torch.nn.LayerNorm(512)
        self.layer_norm2 = torch.nn.LayerNorm(512)

    def forward(self, x: torch.Tensor):
        # x: [batch, tokens, 512]

        # Multi-head attention
        # Self attention, so q, k, v are all x
        x = self.multi_head_attention(q=x, k=x, v=x) + x  # Residual connection
        x = self.layer_norm1(x)

        # Feed forward
        x = self.feed_forward(x) + x  # Residual connection
        x = self.layer_norm2(x)

        return x


class Encoder(torch.nn.Module):
    def __init__(self, num_layers: int = 6):
        super(Encoder, self).__init__()

        layers = [TransformerLayer() for _ in range(num_layers)]
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        # x: [batch, tokens, 512]

        # Transformer layers
        x = self.layers(x)

        return x
