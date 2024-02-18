from typing import Self
from pathlib import Path

from datasets import DatasetDict, Dataset

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.normalizers import BertNormalizer


import learn_former.data.dataset


class CustomTokenizer:
    """Custom Tokenizer for the Transformer model.

    The tokenizer is based on the Hugging Face Tokenizers library and
    can be either trained from a dataset or loaded from a pre-trained
    tokenizer.

    Use `from_pretrained_tokenizer` to load a pre-trained tokenizer from file.
    Use `from_dataset` to train a new tokenizer from a dataset.
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding(pad_id=3, pad_token="[PAD]")

    @classmethod
    def from_pretrained_tokenizer(cls, tokenizer_path: Path) -> Self:
        """Load a pre-trained tokenizer from file.

        Args:
            tokenizer_path (Path): Path to the tokenizer json file.

        Returns:
            CustomTokenizer: Instance of the CustomTokenizer class.
        """
        tokenizer = cls._get_trained_tokenizer(tokenizer_path)
        return cls(tokenizer)

    @classmethod
    def from_dataset(
        cls, dataset_path: Path, dataset_name: str, split: str, language: str
    ) -> Self:
        """Train a new tokenizer from a dataset.

        Args:
            dataset_path (Path): Path to the dataset.
            dataset_name (str): Name of the dataset.
            split (str): Split of the dataset to use for training.
            language (str): Language of the dataset.

        Returns:
            CustomTokenizer: Instance of the CustomTokenizer class.
        """
        dataset: Dataset = learn_former.data.dataset.get_dataset(
            dataset_path, dataset_name, split=split
        )
        dataset: Dataset = dataset["translation"]
        tokenizer = cls._train_tokenizer(dataset, language)
        return cls(tokenizer)

    @staticmethod
    def _get_trained_tokenizer(tokenizer_path: Path) -> Tokenizer:
        tokenizer: Tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return tokenizer

    @staticmethod
    def _train_tokenizer(
        dataset: Dataset,
        language: str,
    ) -> Tokenizer:
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]"])
        tokenizer.pre_tokenizer = BertPreTokenizer()
        tokenizer.normalizer = BertNormalizer()

        def _batch_iterator(dataset, batch_size: int = 1000, language: str = "de"):
            for i in range(0, len(dataset), batch_size):
                data = dataset[i : i + batch_size]
                data = [x[language] for x in data]
                yield data

        tokenizer.train_from_iterator(
            _batch_iterator(dataset, batch_size=1000, language=language),
            trainer=trainer,
            length=len(dataset),
        )
        return tokenizer

    def save_tokenizer(self, tokenizer_path: Path) -> None:
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(tokenizer_path))

    def encode_batch(self, text_batch: list[str]) -> tuple[list, int]:
        output = self.tokenizer.encode_batch(text_batch)

        token_ids = [x.ids for x in output]
        return token_ids

    def decode_batch(self, token_ids: list[int]) -> list[str]:
        output = self.tokenizer.decode_batch(token_ids)
        return output
