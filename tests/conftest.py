"""
Conftest file for the tests directory.
Creates fixtures (functions) that can be used in the tests.
"""

import pytest
import torch
from pathlib import Path
from tokenizers import Tokenizer

import learn_former
import learn_former.model.transformer
from learn_former.data.tokenizer import CustomTokenizer


@pytest.fixture(scope="session")
def learn_former_root_dir() -> Path:
    return Path(learn_former.__file__).parent


@pytest.fixture(scope="session")
def sentence() -> list[str]:
    sentence = "The quick brown fox jumps over the lazy dog. This is a lengthier sentence that can be used for testing purposes."
    return [sentence]


@pytest.fixture(scope="session")
def sentence_batch() -> list[str]:
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "I like cats more. This is a lengthier sentence that can be used for testing purposes.",
    ]
    return sentences


@pytest.fixture(scope="session")
def tokenized_sentence() -> torch.Tensor:
    """Tokenized sentence as tensor (1, 10)."""
    return torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])


@pytest.fixture(scope="session")
def tokenized_sentence_batch() -> torch.Tensor:
    return torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
    )


@pytest.fixture(scope="session")
def embedded_sentence() -> torch.Tensor:
    embeddings = torch.randn(1, 10, 512)
    return embeddings


@pytest.fixture(scope="session")
def embedded_sentence_batch() -> torch.Tensor:
    return torch.randn(2, 10, 512)


@pytest.fixture(scope="session")
def encoder_mask(tokenized_sentence) -> torch.Tensor:
    batch_size, seq_len = tokenized_sentence.shape
    mask = torch.ones(seq_len, seq_len, dtype=bool)

    mask[-3:, :] = False
    mask = mask.expand(batch_size, 1, -1, -1)

    return mask


@pytest.fixture(scope="session")
def encoder_mask_batch(tokenized_sentence_batch) -> torch.Tensor:
    batch_size, seq_len = tokenized_sentence_batch.shape
    mask = torch.ones(seq_len, seq_len, dtype=bool)

    mask[-3:, :] = False
    mask = mask.expand(batch_size, 1, -1, -1)

    return mask


@pytest.fixture(scope="session")
def decoder_mask(tokenized_sentence) -> torch.Tensor:
    batch_size, seq_len = tokenized_sentence.shape
    mask = torch.ones(seq_len, seq_len, dtype=bool)

    mask = torch.tril(mask)
    mask = mask.expand(batch_size, 1, -1, -1)

    return mask


@pytest.fixture(scope="session")
def decoder_mask_batch(tokenized_sentence_batch) -> torch.Tensor:
    batch_size, seq_len = tokenized_sentence_batch.shape
    mask = torch.ones(seq_len, seq_len, dtype=bool)

    mask = torch.tril(mask)
    mask = mask.expand(batch_size, 1, -1, -1)

    return mask


@pytest.fixture(scope="session")
def tokenizer(learn_former_root_dir: Path) -> CustomTokenizer:
    path = learn_former_root_dir.parent / "tests/reference_data/de_tokenizer.json"
    tokenizer = CustomTokenizer.from_pretrained_tokenizer(path)

    return tokenizer


@pytest.fixture(scope="session")
def model(tokenizer: CustomTokenizer) -> torch.nn.Module:

    vocab_size = tokenizer.tokenizer.get_vocab_size()

    model = learn_former.model.transformer.Transformer(
        num_layers=2,
        d_model=128,
        num_heads=4,
        d_ff=256,
        dropout=0.1,
        vocab_size_de=vocab_size,
        vocab_size_en=vocab_size,  # Must be higher than the vocab size of the tokenizer!
        padding_id=0,
    )
    return model
