import torch

from learn_former.model.submodels import (
    MultiHeadAttention,
    FeedForward,
)


class TransformerLayer(torch.nn.Module):
    def __init__(
        self,
        num_heads: int = 8,
        d_model: int = 512,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(
            input_dim=d_model, num_heads=num_heads
        )
        self.feed_forward = FeedForward(input_dim=d_model, hidden_dim=d_ff)

        self.layer_norm1 = torch.nn.LayerNorm(d_model)
        self.layer_norm2 = torch.nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, input_mask: torch.Tensor = None):
        # x: [batch, tokens, 512]

        # Multi-head attention
        # Self attention, so q, k, v are all x
        x = (
            self.multi_head_attention(q=x, k=x, v=x, mask=input_mask) + x
        )  # Residual connection
        x = self.layer_norm1(x)

        # Feed forward
        x = self.feed_forward(x) + x  # Residual connection
        x = self.layer_norm2(x)

        return x


class Encoder(torch.nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        num_heads: int = 8,
        d_model: int = 512,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super(Encoder, self).__init__()

        layers = [
            TransformerLayer(
                d_ff=d_ff, d_model=d_model, num_heads=num_heads, dropout=dropout
            )
            for _ in range(num_layers)
        ]
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, input_mask: torch.Tensor = None):
        # x: [batch, tokens, 512]

        # Transformer layers
        x = self.layers(x, input_mask)

        return x
