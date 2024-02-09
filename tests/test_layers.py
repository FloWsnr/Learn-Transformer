import pytest
import torch
from learn_former.model.submodels import (
    FeedForward,
    MultiHeadAttention,
    Tokenizer,
    Embedding,
    PositionalEncoding,
)


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
    assert x.shape == (1, 10, input_dim)


def test_batch_multiheadattention_layer(embedded_sentence_batch: torch.Tensor):
    input_dim = 512
    num_heads = 8
    mha = MultiHeadAttention(input_dim=input_dim, num_heads=num_heads)

    x = mha(embedded_sentence_batch)
    assert x.shape == (2, 10, input_dim)


def test_tokenizer(sentence: list[str]):
    tokenizer = Tokenizer()
    x = tokenizer(sentence)
    assert x.dtype == torch.int64


def test_batch_tokenizer(sentence_batch: list[str]):
    tokenizer = Tokenizer()
    x = tokenizer(sentence_batch)
    assert x.dtype == torch.int64


def test_word_embedding_layer(tokenized_sentence: torch.Tensor):
    vocab_size = 10000
    embedding_dim = 512
    we = Embedding(vocab_size, embedding_dim)
    x = we(tokenized_sentence)

    token_shape = tokenized_sentence.shape
    assert x.shape == (token_shape[0], token_shape[1], embedding_dim)


def test_batch_word_embedding_layer(tokenized_sentence_batch: torch.Tensor):
    vocab_size = 10000
    embedding_dim = 512
    we = Embedding(vocab_size, embedding_dim)
    x = we(tokenized_sentence_batch)

    token_shape = tokenized_sentence_batch.shape
    assert x.shape == (token_shape[0], token_shape[1], embedding_dim)


def test_positional_encoding_layer(embedded_sentence: torch.Tensor):
    max_len = 1024
    embedding_dims = 512
    pe = PositionalEncoding(max_len, embedding_dims)
    x = pe(embedded_sentence)
    assert x.shape == embedded_sentence.shape


def test_batch_positional_encoding_layer(embedded_sentence_batch: torch.Tensor):
    max_len = 1024
    embedding_dims = 512
    pe = PositionalEncoding(max_len, embedding_dims)
    x = pe(embedded_sentence_batch)
    assert x.shape == embedded_sentence_batch.shape
