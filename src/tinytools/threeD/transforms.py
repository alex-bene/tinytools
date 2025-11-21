"""3D transformation utilities.

This module provides helper functions for working with 3D transformations,
specifically integrating with PyTorch3D structures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytorch3d.structures import Meshes
    from pytorch3d.transforms import Transform3d


def transform_meshes(meshes: Meshes, transform: Transform3d, inplace: bool = False) -> Meshes:
    """Apply a Transform3d to a Meshes object."""
    if not inplace:
        meshes = meshes.clone()
    return meshes.update_padded(transform.transform_points(meshes.verts_padded()))
