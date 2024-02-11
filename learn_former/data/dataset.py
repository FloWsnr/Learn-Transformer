from pathlib import Path

from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict


def get_dataset(
    storage_dir: Path,
    dataset_name: str = "wmt16",
):
    dataset: DatasetDict = load_dataset(
        dataset_name, name="de-en", cache_dir=storage_dir
    )
    return dataset


def get_dataloaders(
    storage_dir: Path,
    dataset_name: str = "wmt16",
    batch_size: int = 32,
    num_workers: int = 4,
):
    dataset = get_dataset(storage_dir, dataset_name)
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
