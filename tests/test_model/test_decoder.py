import pytest
import torch
from learn_former.model.decoder import Decoder


def test_decoder_output_shape(embedded_sentence: torch.Tensor):

    # Embedded sentence has shape (batch, tokens, 512)
    # We need to create a padding plus look-ahead mask of shape (batch, tokens, tokens)
    tokens = embedded_sentence.shape[1]
    target_mask = torch.ones(tokens, tokens, dtype=torch.bool)
    target_mask = torch.tril(target_mask)

    # and a padding mask of shape (batch, tokens)
    input_mask = torch.ones(embedded_sentence.shape[:2], dtype=torch.bool)

    num_layers = 6
    decoder = Decoder(num_layers=num_layers)
    x = decoder(
        embedded_sentence,
        encoder_output=embedded_sentence,
        input_mask=input_mask,
        target_mask=target_mask,
    )
    assert x.shape == embedded_sentence.shape
