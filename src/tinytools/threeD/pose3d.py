"""Module defining a Pose3D dataclass to hold 3D pose data."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tinytools.imports import module_from_obj

try:
    from tensordict.tensorclass import tensorclass as dtclass  # pyright: ignore[reportMissingImports]
except ImportError:
    from dataclasses import dataclass as dtclass


if TYPE_CHECKING:
    from numpy import ndarray
    from torch import Tensor  # pyright: ignore[reportMissingImports]


def is_orthogonal(matrix: ndarray | Tensor, atol: float = 1e-6) -> bool:
    """Check if a matrix is orthogonal.

    Args:
        matrix: The matrix to check.
        atol: Absolute tolerance for floating point comparisons

    """
    n = matrix.shape[0]
    vector_lib = module_from_obj(matrix)
    identity = vector_lib.eye(n, dtype=matrix.dtype)
    if hasattr(identity, "device"):
        identity = identity.to(matrix.device)
    return vector_lib.allclose(matrix @ matrix.mT, identity, atol=atol)


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

    def get_matrix(self) -> ndarray | Tensor:
        """Return the 4x4 transformation matrix representing the pose.

        Returns:
            ndarray | Tensor: (..., 4, 4) transformation matrices.

        """
        batch_dims = self.rotation.shape[:-2]
        vector_lib = module_from_obj(self.rotation)
        matrix = vector_lib.eye(4, dtype=self.rotation.dtype)
        if hasattr(self.rotation, "device"):
            matrix = matrix.to(self.rotation.device)
        if batch_dims:
            matrix = vector_lib.tile(matrix, (*batch_dims, 1, 1))

        matrix[..., :3, :3] = self.rotation
        matrix[..., :3, 3] = self.translation

        return matrix

    def change_basis(self, conversion_matrix: ndarray | Tensor, inplace: bool = False) -> Pose3D | None:
        """Transform the pose into a new coordinate system using a change-of-basis matrix.

        This method applies a similarity transformation to the rotation and transforms the translation vector into the
        new basis.

        Args:
            conversion_matrix: A (3, 3) matrix representing the change of basis.
            inplace: If True, modifies the current instance. If False, returns a new Pose3D instance.

        Returns:
            A new Pose3D instance represented in the new coordinate system.

        Raises:
            ValueError: If the conversion_matrix is not of shape (3, 3).

        """
        if conversion_matrix.shape != (3, 3):
            msg = f"Conversion matrix must have shape (3, 3), but got {conversion_matrix.shape}"
            raise ValueError(msg)
        if not is_orthogonal(conversion_matrix):
            msg = "Conversion matrix must be orthogonal."
            raise ValueError(msg)

        rotation = conversion_matrix @ self.rotation @ conversion_matrix.mT
        translation = self.translation @ conversion_matrix
        if not inplace:
            return Pose3D(rotation=rotation, translation=translation)

        self.rotation = rotation
        self.translation = translation
        return None
