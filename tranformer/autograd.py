"""Example of custom autograd function"""

import torch


class NewActivationFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        result = torch.exp(-(x**2))
        ctx.save_for_backward(x)
        return result

    @staticmethod
    def backward(ctx, grad_outputs: torch.Tensor) -> torch.Tensor:
        (input,) = ctx.saved_tensors
        return -2 * input * torch.exp(-(input**2))


x = torch.rand(21, 8, requires_grad=True)
y = NewActivationFun.apply(x).mean()

y.backward()
