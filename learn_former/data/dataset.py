from pathlib import Path

from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


def get_dataset(
    storage_dir: Path,
    dataset_name: str = "wmt16",
) -> DatasetDict:
    dataset: DatasetDict = load_dataset(
        dataset_name, name="de-en", cache_dir=storage_dir
    )
    return dataset


def get_dataloaders(
    dataset: DatasetDict,
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        dataset["train"], batch_size=batch_size, num_workers=num_workers
    )
    val_loader = DataLoader(
        dataset["validation"], batch_size=batch_size, num_workers=num_workers
    )
    test_loader = DataLoader(
        dataset["test"], batch_size=batch_size, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader


class Tokenizer_Pipeline:
    def __init__(self, storage_dir: Path, dataset_name: str = "wmt16"):
        self.tokenizer = None

        if (storage_dir / "tokenizer.json").exists():
            self.tokenizer = self.load_tokenizer(storage_dir)
            print("Tokenizer loaded from storage")
        else:
            self.dataset = get_dataset(storage_dir, dataset_name)
            self.tokenizer = self.train_tokenizer()
            self.save_tokenizer(self.tokenizer, storage_dir)
            print("Tokenizer trained and saved")

        self.tokenizer.enable_padding(pad_id=3, pad_token="[PAD]")

    def train_tokenizer(self, vocab_size: int = 32000) -> Tokenizer:
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]"])

        tokenizer.pre_tokenizer = Whitespace()

        tokenizer.train_from_iterator(
            self.dataset["train"]["translation.de"],
            trainer=trainer,
            vocab_size=vocab_size,
        )
        return tokenizer

    def save_tokenizer(self, storage_dir: Path) -> None:
        self.tokenizer.save(storage_dir / "tokenizer.json")

    def load_tokenizer(self, storage_dir: Path) -> Tokenizer:
        tokenizer = Tokenizer.from_file(storage_dir / "tokenizer.json")
        return tokenizer

    def encode_batch(self, text_batch: list[str]) -> list[int]:
        output = self.tokenizer.encode_batch(text_batch)

        token_ids = [x.ids for x in output]
        attention_masks = [x.attention_mask for x in output]

        return token_ids, attention_masks

    def decode_batch(self, token_ids: list[int]) -> list[str]:
        output = self.tokenizer.decode_batch(token_ids)
        return output
