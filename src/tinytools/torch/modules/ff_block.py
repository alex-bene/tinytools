"""Pre-Normalization Block Module with Residual Connection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal

import torch.nn.functional as F  # pyright: ignore[reportMissingImports]
from torch import nn  # pyright: ignore[reportMissingImports]

from .gated_mlp import GatedMLP
from .vanilla_mlp import VanillaMLP

if TYPE_CHECKING:
    import torch  # pyright: ignore[reportMissingImports]


class FFBlock(nn.Module):
    """Pre-Normalization Block with optional Residual Connection.

    Applies normalization before passing the input through the given function/module.
    Optionally adds a residual connection.

    Args:
        input_dim (int): Input feature dimension for the normalization layer.
        hidden_dim (int): Hidden layer dimension for the MLP.
        output_dim (int | None, optional): Output feature dimension for the MLP. If None, defaults to input_dim.
            Defaults to None.
        bias (bool, optional): Whether to use bias in linear layers of the MLP. Defaults to True.
        dropout (float, optional): Dropout probability in the MLP. Defaults to 0.0.
        mlp_type (Literal["gated", "vanilla"], optional): Type of MLP to use ("gated" or "vanilla").
            Defaults to "gated".
        activation_fn (Callable[[torch.Tensor], torch.Tensor], optional): Activation function for the MLP.
            Defaults to SiLU.
        norm_first (bool, optional): Whether to apply normalization before the MLP or in the end. Defaults to False.
        norm_fn (Callable[[int], nn.Module], optional): A callable that returns a normalization layer given the input
            dimension. Defaults to nn.LayerNorm.
        residual (bool, optional): Whether to include a residual connection. Defaults to True.

    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        bias: bool = True,
        dropout: float = 0.0,
        mlp_type: Literal["gated", "vanilla"] = "gated",
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        norm_first: bool = False,
        norm_fn: Callable[[int], nn.Module] = nn.LayerNorm,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.norm_first = norm_first
        self.norm = norm_fn(input_dim)
        mlp_kwargs = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim if output_dim is not None else input_dim,
            "bias": bias,
            "dropout": dropout,
            "activation_fn": activation_fn,
        }
        self.mlp = (
            GatedMLP(**mlp_kwargs, skip_output_layer=False)
            if mlp_type == "gated"
            else VanillaMLP(**mlp_kwargs, dropout_at_end=bool(residual))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the PreNormBlock."""
        out = self.norm(x) if self.norm_first else x
        out = self.mlp(out)
        out = x + out if self.residual else out
        return self.norm(out) if not self.norm_first else out
