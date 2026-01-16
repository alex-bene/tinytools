"""Gated MLP module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch.nn.functional as F  # pyright: ignore[reportMissingImports]
from torch import nn  # pyright: ignore[reportMissingImports]

if TYPE_CHECKING:
    import torch  # pyright: ignore[reportMissingImports]


class GatedMLP(nn.Module):
    """Gated MLP layer.

    Support any activation function to get SwiGLU, ReGLU, etc. Default is SiLU for SwiGLU.

    Args:
        input_dim (int): Input feature dimension.
        hidden_dim (int): Hidden layer dimension (typically larger than input_dim).
        output_dim (int): Output feature dimension.
        bias (bool, optional): Whether to use bias in linear layers. Defaults to True.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        skip_output_layer (bool, optional): Whether to skip the output layer (i.e., use identity). Defaults to False.
        activation_fn (Callable[[torch.Tensor], torch.Tensor], optional): Activation function to use. Defaults to SiLU.

    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        bias: bool = True,
        dropout: float = 0.0,
        skip_output_layer: bool = False,
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
    ) -> None:
        super().__init__()
        self.activation = activation_fn
        self.dropout = nn.Dropout(p=dropout)
        self.w1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, output_dim, bias=bias) if not skip_output_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """GLU forward pass."""
        x = self.activation(self.w1(x)) * self.w2(x)
        return self.dropout(self.w3(x))
