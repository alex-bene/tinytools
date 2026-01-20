"""3D transformation utilities.

This module provides helper functions for working with 3D transformations,
specifically integrating with PyTorch3D structures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from tinytools.imports import requires

try:
    from pytorch3d.transforms import Transform3d as pt3d_Transform3d  # pyright: ignore[reportMissingImports]
except ImportError:
    pt3d_Transform3d = None  # type: ignore[assignment]  # noqa: N816
try:
    import torch  # pyright: ignore[reportMissingImports]
except ImportError:
    torch = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from pytorch3d.structures import Meshes  # pyright: ignore[reportMissingImports]
    from pytorch3d.transforms import Transform3d  # pyright: ignore[reportMissingImports]
    from torch import Tensor


class DecomposedTransform(NamedTuple):  # noqa: D101
    scale: Tensor
    rotation: Tensor
    translation: Tensor


def transform_meshes(meshes: Meshes, transform: Transform3d, inplace: bool = False) -> Meshes:
    """Apply a Transform3d to a Meshes object."""
    if not inplace:
        meshes = meshes.clone()
    return meshes.update_padded(transform.transform_points(meshes.verts_padded()))


def compose_transform(scale: Tensor, rotation: Tensor, translation: Tensor) -> Transform3d:
    """Composes a Transform3d from scale, rotation, and translation.

    Args:
        scale: (B, 3) tensor of scale factors
        rotation: (B, 3, 3) tensor of rotation matrices
        translation: (B, 3) tensor of translation vectors

    """
    requires("pytorch3d", "compose_transform requires pytorch3d to be installed.")
    tfm = pt3d_Transform3d(dtype=scale.dtype, device=scale.device)
    return tfm.scale(scale).rotate(rotation).translate(translation)


def decompose_transform(transform: Transform3d) -> DecomposedTransform:
    """Decomposes a Transform3d into scale, rotation, and translation.

    Returns:
        scale: (B, 3) tensor of scale factors
        rotation: (B, 3, 3) tensor of rotation matrices
        translation: (B, 3) tensor of translation vectors

    """
    matrices = transform.get_matrix()
    scale = torch.norm(matrices[:, :3, :3], dim=-1)
    rotation = matrices[:, :3, :3] / scale.unsqueeze(-1)  # Normalize rotation matrix
    translation = matrices[:, 3, :3]  # Extract translation vector
    return DecomposedTransform(scale, rotation, translation)
