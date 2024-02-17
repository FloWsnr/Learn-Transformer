import pytest
import torch

from learn_former.model.transformer import Transformer


def test_transformer_single(tokenized_sentence: torch.Tensor):
    d_model = 512
    model = Transformer(d_model=d_model)

    input = tokenized_sentence
    target = tokenized_sentence

    output = model(input, target)
    assert output.shape == (
        input.shape[0],
        input.shape[1],
        d_model,
    )


def test_transformer_batch(tokenized_sentence_batch: torch.Tensor):
    d_model = 512
    model = Transformer(d_model=d_model)

    input = tokenized_sentence_batch
    target = tokenized_sentence_batch

    output = model(input, target)
    assert output.shape == (
        input.shape[0],
        input.shape[1],
        d_model,
    )
