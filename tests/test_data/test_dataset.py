import pytest
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from learn_former.data.dataset import get_dataloaders, get_dataset


def test_get_split_dataset(learn_former_root_dir: Path):

    dataset_dir = learn_former_root_dir / "data/datasets"
    dataset = get_dataset(storage_dir=dataset_dir, dataset_name="wmt16", split="test")

    x = dataset["translation"][0]
    x_de = x["de"]
    x_en = x["en"]
    assert isinstance(x_de, str)


def test_get_dataloaders(learn_former_root_dir: Path):

    dataset_dir = learn_former_root_dir / "data/datasets"
    dataset = get_dataset(storage_dir=dataset_dir, dataset_name="wmt16")

    dataloaders = get_dataloaders(dataset, batch_size=1, num_workers=0)
    train_loader, val_loader, test_loader = dataloaders

    for batch in val_loader:
        x = batch["de"]
        y = batch["en"]

        assert isinstance(x, list)
        assert isinstance(y, list)

        break
