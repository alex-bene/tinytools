"""Module defining a Pose3D dataclass to hold 3D pose data."""

from __future__ import annotations

from typing import TYPE_CHECKING

try:
    from tensordict.tensorclass import tensorclass as dtclass  # pyright: ignore[reportMissingImports]
except ImportError:
    from dataclasses import dataclass as dtclass


if TYPE_CHECKING:
    from numpy import ndarray
    from torch import Tensor  # pyright: ignore[reportMissingImports]


@dtclass
class Pose3D:
    """A dataclass to hold 3D pose data consisting of rotation matrices and translation vectors.

    Supports both PyTorch Tensors and NumPy ndarrays. If tensordict is installed, this class is a tensorclass,
    allowing for efficient tensor operations and integration with tensordict-based workflows.
    """

    rotation: ndarray | Tensor  # (..., 3, 3) Rotation matrices
    translation: ndarray | Tensor  # (..., 3) Translation vectors

    def __post_init__(self) -> None:
        """Check that the inputs are valid and set batch_size and device possible."""
        if self.rotation.shape[-2:] != (3, 3):
            msg = f"Rotation matrices R must have shape (..., 3, 3), but got {self.rotation.shape}"
            raise ValueError(msg)
        if self.translation.shape[-1] != 3:
            msg = f"Translation vectors T must have shape (..., 3), but got {self.translation.shape}"
            raise ValueError(msg)
        if self.rotation.shape[:-2] != self.translation.shape[:-1]:
            msg = (
                f"Batch dimensions of R and T must match, but got {self.rotation.shape[:-2]} and "
                f"{self.translation.shape[:-1]}"
            )
            raise ValueError(msg)

        if hasattr(self, "auto_batch_size_"):
            self.auto_batch_size_(self.rotation.ndim - 2)
        if hasattr(self, "auto_device_") and self.device is None:
            self.auto_device_()
