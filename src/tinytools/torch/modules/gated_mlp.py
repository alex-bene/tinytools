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
        dropout (float, optional): Dropout probability. Defaults to 0.1.
        skip_output_layer (bool, optional): Whether to skip the output layer (i.e., use identity). Defaults to False.
        dropout_at_end (bool, optional): Whether to apply dropout after the final layer. If use in a residual block,
            then set to true. Otherwise, set to false to avoid running a normalization layer after dropout.
            Defaults to True.
        activation_fn (Callable[[torch.Tensor], torch.Tensor], optional): Activation function to use. Defaults to SiLU.

    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        bias: bool = True,
        dropout: float = 0.1,
        skip_output_layer: bool = False,
        dropout_at_end: bool = True,
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
    ) -> None:
        super().__init__()
        if not dropout_at_end and skip_output_layer:
            msg = (
                "If skip_output_layer is True and dropout is to be applied, it will be applied at the end so that "
                "`dropout_at_end==False` and `skip_output_layer==True` is invalid."
            )
            raise ValueError(msg)
        self.activation = activation_fn
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_at_end = dropout_at_end
        self.w1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, output_dim, bias=bias) if not skip_output_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """GLU forward pass."""
        x = self.activation(self.w1(x)) * self.w2(x)
        x = self.dropout(x) if not self.dropout_at_end else x
        return self.dropout(self.w3(x)) if self.dropout_at_end else self.w3(x)
