import torch

from learn_former.model.submodels import (
    MultiHeadAttention,
    FeedForward,
)


class TransformerDecoderLayer(torch.nn.Module):
    def __init__(
        self,
        num_heads: int = 8,
        d_model: int = 512,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super(TransformerDecoderLayer, self).__init__()
        # Self attention for decoder input
        self.multi_head_self_attention = MultiHeadAttention(
            num_heads=num_heads, input_dim=d_model, dropout=dropout
        )

        # Cross attention for encoder-decoder
        self.multi_head_cross_attention = MultiHeadAttention(
            num_heads=num_heads, input_dim=d_model, dropout=dropout
        )

        self.feed_forward = FeedForward(input_dim=d_model, hidden_dim=d_ff)

        self.layer_norm1 = torch.nn.LayerNorm(d_model)
        self.layer_norm2 = torch.nn.LayerNorm(d_model)
        self.layer_norm3 = torch.nn.LayerNorm(d_model)

    def forward(
        self,
        decoder_input: torch.Tensor,
        encoder_output: torch.Tensor,
        input_mask: torch.Tensor = None,
        target_mask: torch.Tensor = None,
    ):
        # decoder_input: [batch, tokens, 512]

        # Multi-head attention
        # Self attention, so q, k, v are all decoder_input
        # Mask is the mask of the target (decoder_input), since we
        # don't want to attend to future tokens
        x = (
            self.multi_head_self_attention(
                q=decoder_input,
                k=decoder_input,
                v=decoder_input,
                mask=target_mask,
            )
            + decoder_input
        )  # Residual connection
        x = self.layer_norm1(x)

        # Cross attention
        # q is decoder_input, k and v are encoder_output
        # Mask is the mask of the input (encoder), since we
        # want to attend to all input tokens for each target token
        x = (
            self.multi_head_cross_attention(
                q=x,
                k=encoder_output,
                v=encoder_output,
                mask=input_mask,
            )
            + x
        )
        x = self.layer_norm2(x)

        # Feed forward
        x = self.feed_forward(x) + x  # Residual connection
        x = self.layer_norm3(x)

        return x


class Decoder(torch.nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        num_heads: int = 8,
        d_model: int = 512,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super(Decoder, self).__init__()

        layers = [
            TransformerDecoderLayer(num_heads, d_model, d_ff, dropout)
            for _ in range(num_layers)
        ]
        self.layers = torch.nn.ModuleList(layers)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        input_mask: torch.Tensor = None,
        target_mask: torch.Tensor = None,
    ):
        # x: [batch, tokens, 512]

        # Transformer layers
        for layer in self.layers:
            x = layer(x, encoder_output, input_mask=input_mask, target_mask=target_mask)

        return x
