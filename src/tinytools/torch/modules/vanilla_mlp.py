"""Vanilla MLP module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch.nn.functional as F  # pyright: ignore[reportMissingImports]
from torch import nn  # pyright: ignore[reportMissingImports]

if TYPE_CHECKING:
    import torch  # pyright: ignore[reportMissingImports]


class VanillaMLP(nn.Module):
    """Vanilla MLP layer.

    Supports any activation function. Default is ReLU.

    Args:
        input_dim (int): Input feature dimension.
        hidden_dim (int): Hidden layer dimension (typically larger than input_dim).
        output_dim (int): Output feature dimension.
        bias (bool): Whether to use bias in linear layers. Defaults to True.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
        dropout_at_end (bool, optional): Whether to apply dropout after the final layer. If use in a residual block,
            then set to true. Otherwise, set to false to avoid running a normalization layer after dropout.
            Defaults to True
        dropout_at_mid (bool, optional): Whether to apply dropout after the hidden layer. Defaults to True.
        activation_fn (Callable[[torch.Tensor], torch.Tensor], optional): Activation function to use. Defaults to ReLU.

    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        bias: bool = True,
        dropout: float = 0.1,
        dropout_at_end: bool = True,
        dropout_at_mid: bool = True,
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.relu,
    ) -> None:
        super().__init__()
        self.dropout_at_end = dropout_at_end
        self.dropout_at_mid = dropout_at_mid
        self.activation = activation_fn
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Vanilla MLP forward pass."""
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x) if self.dropout_at_mid else x
        x = self.fc2(x)
        return self.dropout(x) if self.dropout_at_end else x
