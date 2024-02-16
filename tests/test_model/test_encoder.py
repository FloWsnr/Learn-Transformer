import pytest
import torch

from learn_former.model.encoder import Encoder


def test_encoder_output_shape(embedded_sentence: torch.Tensor):

    # Embedded sentence has shape (batch, tokens, 512)
    # We need to create a padding mask of shape (batch, tokens)
    padding_mask = torch.ones(embedded_sentence.shape[:2], dtype=torch.bool)
    padding_mask[:, -3:] = False

    num_layers = 6
    encoder = Encoder(num_layers=num_layers)
    x = encoder(embedded_sentence, input_mask=padding_mask)
    assert x.shape == embedded_sentence.shape
