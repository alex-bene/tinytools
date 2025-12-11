"""PyTorch utility functions."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch  # pyright: ignore[reportMissingImports]


def freeze_model(model: "torch.nn.Module") -> None:
    """Freeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = False
