"""PyTorch utility functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tinytools.imports import optional_module

if TYPE_CHECKING:
    import torch  # pyright: ignore[reportMissingImports]

    from tinytools.array_ops import ArrayTensor
else:
    torch = optional_module("torch", extra="torch")


def freeze_module(module: torch.nn.Module) -> None:
    """Freeze all module parameters."""
    for param in module.parameters():
        param.requires_grad = False


def get_zero_safe_values(tensor: torch.Tensor, eps: float) -> torch.Tensor:
    """Replace values in tensor that are close to zero with +/- eps."""
    return torch.where(tensor.abs() < eps, eps * torch.sign(tensor).add(tensor == 0), tensor)


def as_float_tensor(x: ArrayTensor, *, device: torch.device | None) -> torch.Tensor:
    """Convert intrinsics to a tensor."""
    if torch.is_tensor(x):
        x = x.to(device=device)
        if not x.is_floating_point():
            x = x.float()
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def is_integer(x: torch.Tensor) -> bool:
    """Check if a tensor is an integer type."""
    return not (x.is_floating_point() or x.is_complex() or x.dtype == torch.bool)


def as_int_tensor(x: ArrayTensor, *, device: torch.device | None) -> torch.Tensor:
    """Convert intrinsics to a tensor."""
    if torch.is_tensor(x):
        x = x.to(device=device)
        if not is_integer(x):
            x = x.long()
    return torch.as_tensor(x, dtype=torch.long, device=device)
