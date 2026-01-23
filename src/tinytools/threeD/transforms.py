"""3D transformation utilities.

This module provides helper functions for working with 3D transformations,
specifically integrating with PyTorch3D structures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch

from tinytools.imports import requires

try:
    from pytorch3d.transforms import Transform3d as pt3d_Transform3d  # pyright: ignore[reportMissingImports]
    from pytorch3d.transforms import (  # pyright: ignore[reportMissingImports]
        axis_angle_to_matrix,
        matrix_to_axis_angle,
        matrix_to_quaternion,
        matrix_to_rotation_6d,
        quaternion_to_matrix,
        rotation_6d_to_matrix,
    )
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


def broadcast_postcompose(
    scale: Tensor, rotation: Tensor, translation: Tensor, transform_to_postcompose: Transform3d
) -> tuple[Tensor, Tensor, Tensor]:
    """Broadcasted post-composition of a transform defined by scale, rotation, translation with another transform.

    Args:
        scale(torch.Tensor): (B, ..., 1 or 3) tensor of scale factors
        rotation(torch.Tensor): (B, ..., 4 or 3 or 6 or *(3, 3)) tensor of rotations
        translation(torch.Tensor): (B, ..., 3) tensor of translation vectors
        transform_to_postcompose(Transform3d): transform to post-compose with. Should have shape (B, 4, 4)

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Decomposed scale, rotation, translation after
            post-composition.

    """
    rotation_repr = "mat"
    if rotation is None:
        rotation_repr = None
        rotation = torch.eye(3, device=translation.device).expand(*translation.shape[:-1], 3, 3)
    if rotation.shape[-1] == 4:  # quaternion
        rotation = quaternion_to_matrix(rotation)
        rotation_repr = "quaternion"
    if rotation.shape[-1] == 6:  # 6D
        rotation = rotation_6d_to_matrix(rotation)
        rotation_repr = "rotation_6d"
    if rotation.shape[-1] == 3 and rotation.ndim == translation.ndim:  # axis-angle
        rotation = axis_angle_to_matrix(rotation)
        rotation_repr = "axis_angle"

    single_scale = scale.shape[-1] == 1
    if single_scale:
        scale = scale.expand(*scale.shape[:-1], 3)

    b = scale.shape[0]
    lead_dims = scale.shape[:-1]
    flattened_lead_dims_size = int(torch.prod(torch.tensor(lead_dims)).item())
    # Create transform of shape (flattened_lead_dims_size)
    composed = compose_transform(
        scale=scale.reshape(flattened_lead_dims_size, 3),
        rotation=rotation.reshape(flattened_lead_dims_size, 3, 3),
        translation=translation.reshape(flattened_lead_dims_size, 3),
    )

    # Apply transform to shape (flattened_lead_dims_size)
    pc_transform: Tensor = transform_to_postcompose.get_matrix()  # size B, 4, 4
    pc_transform = pc_transform.repeat(flattened_lead_dims_size // b, 1, 1)  # size (B * K, 4, 4)
    stacked_pc_transform = pt3d_Transform3d(matrix=pc_transform)
    postcomposed = composed.compose(stacked_pc_transform)

    # Decompose back to shape (B, ..., C)
    scale, rotation, translation = decompose_transform(postcomposed)
    scale = scale.reshape(*lead_dims, 3)
    rotation = rotation.reshape(*lead_dims, 3, 3)
    translation = translation.reshape(*lead_dims, 3)
    if single_scale:
        scale = scale[..., 0].unsqueeze(-1)
    if rotation_repr == "quaternion":
        rotation = matrix_to_quaternion(rotation)
    if rotation_repr == "axis_angle":
        rotation = matrix_to_axis_angle(rotation)
    if rotation_repr == "rotation_6d":
        rotation = matrix_to_rotation_6d(rotation)
    if rotation_repr is None:
        rotation = None
    return scale, rotation, translation
