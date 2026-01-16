"""PyTorch utility functions."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch  # pyright: ignore[reportMissingImports]


def freeze_module(module: "torch.nn.Module") -> None:
    """Freeze all module parameters."""
    for param in module.parameters():
        param.requires_grad = False
