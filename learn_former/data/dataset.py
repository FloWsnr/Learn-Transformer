from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k


def get_dataloaders(
    root,
    splits: tuple[str] = ("train", "val", "test"),
    language_pair: tuple[str] = ("en", "de"),
    batch_size: int = 64,
    num_workers: int = 4,
):
    train, val, test = Multi30k(root=root, split=splits, language_pair=language_pair)
    return (
        DataLoader(train, batch_size=batch_size, num_workers=num_workers),
        DataLoader(val, batch_size=batch_size, num_workers=num_workers),
        DataLoader(test, batch_size=batch_size, num_workers=num_workers),
    )
