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
        name="tr-en",
        cache_dir=storage_dir,
        split=split,
    )
    return dataset


def get_dataloaders(
    dataset: DatasetDict,
    batch_size: int = 32,
    num_workers: int = 4,
    slice: str = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = dataset["train"]["translation"]
    val_dataset = dataset["validation"]["translation"]
    test_dataset = dataset["test"]["translation"]

    if slice is not None:
        train_dataset = train_dataset[:slice]
        val_dataset = val_dataset[:slice]
        test_dataset = test_dataset[:slice]

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader
