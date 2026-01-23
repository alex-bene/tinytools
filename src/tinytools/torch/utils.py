"""PyTorch utility functions."""

import torch  # pyright: ignore[reportMissingImports]


def freeze_module(module: "torch.nn.Module") -> None:
    """Freeze all module parameters."""
    for param in module.parameters():
        param.requires_grad = False


def get_zero_safe_values(tensor: torch.Tensor, eps: float) -> torch.Tensor:
    """Replace values in tensor that are close to zero with +/- eps."""
    return torch.where(tensor.abs() < eps, eps * torch.sign(tensor).add(tensor == 0), tensor)
