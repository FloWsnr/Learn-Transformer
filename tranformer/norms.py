import torch


class LayerNorm(torch.nn.Module):
    def __init__(self, shape: tuple[int, ...], eps: float = 1e-8) -> None:
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(*shape))
        self.beta = torch.nn.Parameter(torch.zeros(*shape))

        self.eps = eps
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(-len(self.shape), 0))
        var, mean = torch.var_mean(x, dim=dims, keepdim=True, correction=0)

        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.gamma + self.beta

        return x


class RMSNorm(torch.nn.Module):
    def __init__(self, shape: tuple[int, ...], eps: float = 1e-8) -> None:
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(*shape))
        self.dims = tuple(
            range(
                -len(
                    shape,
                ),
                0,
            )
        )
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=self.dims, keepdim=True)

        x = x / torch.sqrt(norm + self.eps) * self.gamma

        return x


class GroupNorm(torch.nn.Module):
    def __init__(self, n_groups: int, n_channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(n_channels))
        self.beta = torch.nn.Parameter(torch.zeros(n_channels))
        self.eps = eps
        self.n_groups = n_groups
        self.n_channels = n_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if c != self.n_channels:
            raise ValueError("Input channels must match n_channels.")
        if c % self.n_groups != 0:
            raise ValueError("n_channels must be divisible by n_groups.")
        c_g = c // self.n_groups

        x = x.view(b, self.n_groups, c_g, h, w)
        var, mu = torch.var_mean(x, (2, 3, 4), keepdim=True, correction=0)
        x = (x - mu) / torch.sqrt(var + self.eps)

        x = x.view(b, c, h, w)
        x = x * self.gamma.view(1, c, 1, 1) + self.beta.view(1, c, 1, 1)

        return x


if __name__ == "__main__":
    x = torch.randn(16, 8, 32)
    norm = LayerNorm(shape=(8, 32))

    x = norm(x)
