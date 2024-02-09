import torch

from learn_former.model.submodels import (
    MultiHeadAttention,
    FeedForward,
)


class TransformerDecoderLayer(torch.nn.Module):
    def __init__(self):
        super(TransformerDecoderLayer, self).__init__()
        # Self attention for decoder input
        self.multi_head_self_attention = MultiHeadAttention()

        # Cross attention for encoder-decoder
        self.multi_head_cross_attention = MultiHeadAttention()

        self.feed_forward = FeedForward()

        self.layer_norm1 = torch.nn.LayerNorm(512)
        self.layer_norm2 = torch.nn.LayerNorm(512)
        self.layer_norm3 = torch.nn.LayerNorm(512)

    def forward(self, decoder_input: torch.Tensor, encoder_output: torch.Tensor):
        # decoder_input: [batch, tokens, 512]

        # Multi-head attention
        # Self attention, so q, k, v are all decoder_input
        x = (
            self.multi_head_self_attention(decoder_input, decoder_input, decoder_input)
            + decoder_input
        )  # Residual connection
        x = self.layer_norm1(x)

        # Cross attention
        # q is decoder_input, k and v are encoder_output
        x = self.multi_head_cross_attention(x, encoder_output, encoder_output) + x
        x = self.layer_norm2(x)

        # Feed forward
        x = self.feed_forward(x) + x  # Residual connection
        x = self.layer_norm3(x)

        return x


class Decoder(torch.nn.Module):
    def __init__(self, num_layers: int = 6):
        super(Decoder, self).__init__()

        layers = [TransformerDecoderLayer() for _ in range(num_layers)]
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        # x: [batch, tokens, 512]

        # Transformer layers
        x = self.layers(x)

        return x
