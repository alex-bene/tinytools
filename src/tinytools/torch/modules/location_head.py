"""Location head module predicting 3D locations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tinytools.imports import optional_attr, optional_module

if TYPE_CHECKING:
    import torch  # pyright: ignore[reportMissingImports]
    from torch import nn  # pyright: ignore[reportMissingImports]
else:
    torch = optional_module("torch", extra="torch")
    nn = optional_attr("torch", "nn", extra="torch")


class LocationHead(nn.Module):
    """Location head module predicting 3D locations.

    Note that the xy-head is different from the depth-head as these usually have different representations.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.xy_head = nn.Linear(in_features=input_dim, out_features=2)
        self.depth_head = nn.Linear(in_features=input_dim, out_features=1)

    if TYPE_CHECKING:

        def __call__(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
            """Type hinting fix."""
            return self.forward(hidden_states=hidden_states)

    def forward(
        self, hidden_states: torch.Tensor, hidden_states_depth: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Forward location head."""
        return {
            "xy": self.xy_head(hidden_states),
            "depth": self.depth_head(hidden_states_depth if hidden_states_depth is not None else hidden_states),
        }
