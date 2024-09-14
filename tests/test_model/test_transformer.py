import pytest
import torch

from learn_former.model.transformer import Transformer


def test_transformer_single(tokenized_sentence: torch.Tensor):
    d_model = 512
    vocab_size = 100

    model = Transformer(
        d_model=d_model, vocab_size_de=vocab_size, vocab_size_en=vocab_size
    )

    input = tokenized_sentence
    target = tokenized_sentence

    output = model(input, target)
    assert output.shape == (
        target.shape[0],
        target.shape[1],
        vocab_size,
    )


def test_transformer_batch(tokenized_sentence_batch: torch.Tensor):
    d_model = 512
    vocab_size = 100

    model = Transformer(
        d_model=d_model, vocab_size_de=vocab_size, vocab_size_en=vocab_size
    )

    input = tokenized_sentence_batch
    target = tokenized_sentence_batch

    output = model(input, target)
    assert output.shape == (
        target.shape[0],
        target.shape[1],
        vocab_size,
    )


def test_transformer_with_different_parameters(tokenized_sentence: torch.Tensor):
    d_model = 128
    vocab_size = 100
    num_layers = 2
    num_heads = 4
    d_ff = 256
    dropout = 0.1
    padding_id = 8

    model = Transformer(
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout,
        padding_id=padding_id,
        d_model=d_model,
        vocab_size_de=vocab_size,
        vocab_size_en=vocab_size,
    )

    input = tokenized_sentence
    target = tokenized_sentence

    output = model(input, target)
    assert output.shape == (
        target.shape[0],
        target.shape[1],
        vocab_size,
    )


def test_transformer_batch_difference_in_token_num(
    tokenized_sentence_batch: torch.Tensor,
):
    d_model = 512
    vocab_size = 100

    model = Transformer(
        d_model=d_model, vocab_size_de=vocab_size, vocab_size_en=vocab_size
    )

    input = tokenized_sentence_batch

    # Remove the last token from the target
    target = tokenized_sentence_batch[:, :-1]

    output = model(input, target)
    assert output.shape == (
        target.shape[0],
        target.shape[1],
        vocab_size,
    )


def test_transformer_with_conftest_model(model: Transformer):
    input = torch.randint(0, 100, (16, 10))
    target = torch.randint(0, 100, (16, 9))

    output = model(input, target)
    assert output.shape == (target.shape[0], target.shape[1], model.vocab_size)


def test_transformer_output_is_not_nan(model: Transformer):
    input = torch.randint(0, 100, (16, 10))
    target = torch.randint(0, 100, (16, 9))

    output = model(input, target)
    assert not torch.isnan(output).any()


def test_transformer_cuda(model: Transformer):
    device = "cuda"
    model.to(device)
    input = torch.randint(0, 100, (16, 10)).to(device)
    target = torch.randint(0, 100, (16, 9)).to(device)

    output = model(input, target)
    assert output.shape == (
        target.shape[0],
        target.shape[1],
        model.vocab_size,
    )
