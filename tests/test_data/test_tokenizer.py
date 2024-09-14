import pytest
from pathlib import Path

from learn_former.data.tokenizer import CustomTokenizer


def test_tokenizer_from_file(learn_former_root_dir: Path):
    tokenizer_path = (
        learn_former_root_dir.parent / r"tests\reference_data\de_tokenizer.json"
    )

    tokenizer = CustomTokenizer.from_pretrained_tokenizer(tokenizer_path)
    assert tokenizer is not None


def test_tokenizer_encoding_decoding(learn_former_root_dir: Path):
    tokenizer_path = (
        learn_former_root_dir.parent / r"tests\reference_data\de_tokenizer.json"
    )

    tokenizer = CustomTokenizer.from_pretrained_tokenizer(tokenizer_path)

    text = ["Dies ist ein Test", "Das ist ein anderer Test"]
    encoded = tokenizer.encode_batch(text)
    decoded = tokenizer.decode_batch(encoded)

    assert text == decoded


def test_tokenizer_from_dataset(learn_former_root_dir: Path):
    dataset_path = learn_former_root_dir / "data/datasets"
    dataset_name = "wmt16"
    language = "tr"
    split = "test"

    tokenizer = CustomTokenizer.from_dataset(
        dataset_path, dataset_name, split, language
    )
    assert tokenizer is not None
