import torch

from learn_former.model.decoder import Decoder
from learn_former.model.encoder import Encoder
from learn_former.model.submodels import Embedding, PositionalEncoding


class Transformer(torch.nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        vocab_size: int = 10000,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            num_layers=num_layers,
            num_heads=num_heads,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.decoder = Decoder(
            num_layers=num_layers,
            num_heads=num_heads,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.positional_encoding = PositionalEncoding(embedding_dims=d_model)

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        input_mask: torch.Tensor = None,
        target_mask: torch.Tensor = None,
    ):
        # x: [batch, tokens, 512]

        # Embedding
        input = self.embedding(input)
        target = self.embedding(target)

        # Positional encoding
        input = self.positional_encoding(input)
        target = self.positional_encoding(target)

        # Encoder
        enc_output = self.encoder(input, mask=input_mask)

        # Decoder
        x = self.decoder(
            decoder_input=target, encoder_output=enc_output, mask=target_mask
        )

        return x
