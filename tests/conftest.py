"""
Conftest file for the tests directory.
Creates fixtures (functions) that can be used in the tests.
"""

import pytest
import torch
from pathlib import Path

import learn_former


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
def tokenized_sentence_batch():
    return torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]
    )


@pytest.fixture(scope="session")
def embedded_sentence():
    embeddings = torch.randn(1, 10, 512)
    return embeddings


@pytest.fixture(scope="session")
def embedded_sentence_batch():
    return torch.randn(2, 10, 512)
