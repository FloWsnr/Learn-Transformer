import pytest
import torch
from learn_former.model.decoder import Decoder


def test_decoder(
    embedded_sentence: torch.Tensor,
    encoder_mask: torch.Tensor,
    decoder_mask: torch.Tensor,
):
    num_layers = 6
    decoder = Decoder(num_layers=num_layers)
    x = decoder(
        embedded_sentence,
        encoder_output=embedded_sentence,
        input_mask=encoder_mask,
        target_mask=decoder_mask,
    )
    assert x.shape == embedded_sentence.shape


def test_decoder_batch(
    embedded_sentence_batch: torch.Tensor,
    encoder_mask_batch: torch.Tensor,
    decoder_mask_batch: torch.Tensor,
):
    num_layers = 6
    decoder = Decoder(num_layers=num_layers)
    x = decoder(
        embedded_sentence_batch,
        encoder_output=embedded_sentence_batch,
        input_mask=encoder_mask_batch,
        target_mask=decoder_mask_batch,
    )
    assert x.shape == embedded_sentence_batch.shape
