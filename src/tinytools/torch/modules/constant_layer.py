"""A module that defines a ConstantLayer which outputs a constant tensor of a specified shape."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Iterable


class ConstantLayer(nn.Module):
    """A layer that returns a constant tensor of a specified shape.

    Args:
        output_shape (int | Iterable[int]): The shape of the output tensor (excluding batch dimension).
        value (float, optional): The constant value to fill the tensor with. Defaults to 0.0.
        dtype (torch.dtype | None, optional): The data type of the output tensor.
            If None, uses the input tensor's dtype. Defaults to None.

    """

    def __init__(self, output_shape: int | Iterable[int], value: float = 0.0, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        if isinstance(output_shape, int):
            output_shape = (output_shape,)
        self.output_shape = output_shape
        self.value = value
        self.dtype = dtype

    if TYPE_CHECKING:

        def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
            """Type hinting fix."""
            return self.forward(hidden_states=hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward constant layer."""
        dtype = self.dtype if self.dtype is not None else hidden_states.dtype
        return torch.full(
            (*hidden_states.shape[:-1], *self.output_shape),
            fill_value=self.value,
            device=hidden_states.device,
            dtype=dtype,
        )
