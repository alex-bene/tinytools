"""PyTorch utility functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tinytools.imports import optional_module

if TYPE_CHECKING:
    import torch  # pyright: ignore[reportMissingImports]
else:
    torch = optional_module("torch", extra="torch")


def freeze_module(module: torch.nn.Module) -> None:
    """Freeze all module parameters."""
    for param in module.parameters():
        param.requires_grad = False


def get_zero_safe_values(tensor: torch.Tensor, eps: float) -> torch.Tensor:
    """Replace values in tensor that are close to zero with +/- eps."""
    return torch.where(tensor.abs() < eps, eps * torch.sign(tensor).add(tensor == 0), tensor)
