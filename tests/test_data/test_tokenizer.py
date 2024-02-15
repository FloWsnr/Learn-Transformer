import pytest
from pathlib import Path

from learn_former.data.tokenizer import CustomTokenizer


def test_tokenizer(learn_former_root_dir: Path):

    dataset_dir = learn_former_root_dir / "data/datasets"
    tokenizer_path = learn_former_root_dir / "data/tokenizers/tokenizer.json"

    tokenizer = CustomTokenizer(
        tokenizer_path, dataset_dir=dataset_dir, dataset_name="wmt16"
    )

    assert tokenizer.tokenizer is not None
