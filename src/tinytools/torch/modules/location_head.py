"""Location head module predicting 3D locations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal

import torch
from torch import nn
from torch.nn import functional as F

from .ff_block import FFBlock


class LocationHead(nn.Module):
    """Location head module predicting 3D locations.

    Note that the xy-head is different from the depth-head as these usually have different representations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        bias: bool = True,
        dropout: float = 0.1,
        mlp_type: Literal["gated", "vanilla"] = "vanilla",
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        norm_first: bool = True,
        norm_fn: Callable[[int], nn.Module] = nn.LayerNorm,
        residual: bool = False,
    ) -> None:
        super().__init__()
        ff_block_kwargs = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim if hidden_dim is not None else 4 * input_dim,
            "bias": bias,
            "dropout": dropout,
            "dropout_at_end": False,
            "mlp_type": mlp_type,
            "activation_fn": activation_fn,
            "norm_first": norm_first,
            "norm_fn": norm_fn,
            "residual": residual,
        }
        self.xy_head = FFBlock(output_dim=2, **ff_block_kwargs)
        self.depth_head = FFBlock(output_dim=1, **ff_block_kwargs)

    if TYPE_CHECKING:

        def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
            """Type hinting fix."""
            return self.forward(hidden_states=hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward location head."""
        return {"xy": self.xy_head(hidden_states), "depth": self.depth_head(hidden_states)}
