"""Dataset for LLM training"""

import torch


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, shape: tuple[int, ...], len: int) -> None:
        super().__init__()

        self.data = torch.rand(len, *shape)
        self.len = len

    def __getitem__(self, index) -> torch.Tensor:
        return self.data[index, ...]

    def __len__(self) -> int:
        return self.len
