import pytest
import torch
from learn_former.model.submodels import FeedForward, MultiHeadAttention


def test_feedforward_layer(embedded_sentence: torch.Tensor):
    output_dim = 512
    ffm = FeedForward(input_dim=output_dim, hidden_dim=2048)
    x = ffm(embedded_sentence)
    assert x.shape == (10, output_dim)


def test_batch_feedforward_layer(embedded_sentence_batch: torch.Tensor):
    output_dim = 512
    ffm = FeedForward(input_dim=output_dim, hidden_dim=2048)
    x = ffm(embedded_sentence_batch)
    assert x.shape == (2, 10, output_dim)


def test_multiheadattention_layer(embedded_sentence: torch.Tensor):
    input_dim = 512
    num_heads = 8
    mha = MultiHeadAttention(input_dim=input_dim, num_heads=num_heads)

    x = mha(embedded_sentence)
    assert x.shape == (10, input_dim)


def test_batch_multiheadattention_layer(embedded_sentence_batch: torch.Tensor):
    input_dim = 512
    num_heads = 8
    mha = MultiHeadAttention(input_dim=input_dim, num_heads=num_heads)

    x = mha(embedded_sentence_batch)
    assert x.shape == (2, 10, input_dim)


def test_word_embedding_layer():
    pass


def test_positional_encoding_layer():
    pass
