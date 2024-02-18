from pathlib import Path

from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict


def get_dataset(
    storage_dir: Path,
    dataset_name: str = "wmt16",
    split: str = None,
) -> DatasetDict:
    dataset: DatasetDict = load_dataset(
        dataset_name,
        name="de-en",
        cache_dir=storage_dir,
        split=split,
    )
    return dataset


def get_dataloaders(
    dataset: DatasetDict,
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        dataset["train"]["translation"], batch_size=batch_size, num_workers=num_workers
    )
    val_loader = DataLoader(
        dataset["validation"]["translation"],
        batch_size=batch_size,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        dataset["test"]["translation"], batch_size=batch_size, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader
