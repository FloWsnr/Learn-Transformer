import pytest
import torch

from learn_former.model.encoder import Encoder


def test_encoder(embedded_sentence: torch.Tensor, encoder_mask: torch.Tensor):

    num_layers = 6
    encoder = Encoder(num_layers=num_layers)
    x = encoder(embedded_sentence, input_mask=encoder_mask)
    assert x.shape == embedded_sentence.shape


def test_encoder_batch(
    embedded_sentence_batch: torch.Tensor, encoder_mask_batch: torch.Tensor
):
    num_layers = 6
    encoder = Encoder(num_layers=num_layers)
    x = encoder(embedded_sentence_batch, input_mask=encoder_mask_batch)
    assert x.shape == embedded_sentence_batch.shape
