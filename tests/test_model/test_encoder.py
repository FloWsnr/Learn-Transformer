import pytest
import torch

from learn_former.model.encoder import Encoder


def test_encoder_output_shape(embedded_sentence: torch.Tensor):
    num_layers = 6
    encoder = Encoder(num_layers=num_layers)
    x = encoder(embedded_sentence)
    assert x.shape == embedded_sentence.shape

