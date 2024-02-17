import torch

from learn_former.model.decoder import Decoder
from learn_former.model.encoder import Encoder
from learn_former.model.submodels import Embedding, PositionalEncoding, MaskGenerator


class Transformer(torch.nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        vocab_size: int = 10000,
        padding_id: int = 0,
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

        self.embedding = Embedding(num_embeddings=vocab_size, dim_embeddings=d_model)
        self.positional_encoding = PositionalEncoding(embedding_dims=d_model)

        self.dropout = torch.nn.Dropout(dropout)

        self.mask_generator = MaskGenerator(padding_id=padding_id)

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ):
        # x: [batch, tokens]
        # Generate mask for the decoder input and encoder output
        # Add a look-ahead mask to the decoder input
        input_mask = self.mask_generator(input)
        target_mask = self.mask_generator(target, look_ahead=True)

        # Embedding
        input = self.embedding(input)
        target = self.embedding(target)

        # Positional encoding
        input = self.positional_encoding(input)
        target = self.positional_encoding(target)

        # Dropout on the embeddings
        input = self.dropout(input)
        target = self.dropout(target)

        # Encoder
        ############################
        enc_output = self.encoder(input, input_mask=input_mask)

        # Decoder
        ############################
        x = self.decoder(
            target,
            encoder_output=enc_output,
            input_mask=input_mask,
            target_mask=target_mask,
        )

        return x
