from pathlib import Path

from datasets import DatasetDict, Dataset

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


from learn_former.data.dataset import get_dataset, get_dataloaders


class CustomTokenizer:
    def __init__(
        self, tokenizer_path: Path, dataset_dir: Path, dataset_name: str = "wmt16"
    ):
        self.tokenizer = None

        if (tokenizer_path).exists():
            self.tokenizer = self.load_tokenizer(tokenizer_path)
            print("Tokenizer loaded from storage")
        else:
            print("No tokenizer found, training new tokenizer")
            print("For this, the dataset is needed. Downloading...")
            dataset: DatasetDict = get_dataset(dataset_dir, dataset_name)
            self.dataset: Dataset = dataset["test"]["translation"]

            self.tokenizer = self.train_tokenizer()
            self.save_tokenizer(tokenizer_path)
            print("Tokenizer trained and saved")

        self.tokenizer.enable_padding(pad_id=3, pad_token="[PAD]")

    def load_tokenizer(self, tokenizer_path: Path) -> Tokenizer:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return tokenizer

    def encode_batch(self, text_batch: list[str]) -> list[int]:
        output = self.tokenizer.encode_batch(text_batch)

        token_ids = [x.ids for x in output]
        attention_masks = [x.attention_mask for x in output]

        return token_ids, attention_masks

    def decode_batch(self, token_ids: list[int]) -> list[str]:
        output = self.tokenizer.decode_batch(token_ids)
        return output

    def train_tokenizer(self) -> Tokenizer:

        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]"])

        tokenizer.pre_tokenizer = Whitespace()

        tokenizer.train_from_iterator(
            self._batch_iterator(self.dataset, batch_size=1000),
            trainer=trainer,
            length=len(self.dataset),
        )
        return tokenizer

    def save_tokenizer(self, tokenizer_path: Path) -> None:
        self.tokenizer.save(str(tokenizer_path))

    def _batch_iterator(self, dataset, batch_size: int = 1000):
        for i in range(0, len(dataset), batch_size):

            data = dataset[i : i + batch_size]
            data = [x["de"] for x in data]
            yield data
