"""Meshes class for TinyHumans.

This module defines the Meshes and BodyMeshes classes, which extend PyTorch3D's Meshes class to provide additional
functionality for working with 3D meshes, including conversion to Trimesh objects and handling of body-specific
parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tinytools import get_logger

try:
    import torch
    from pytorch3d.renderer import TexturesAtlas, TexturesVertex
    from trimesh import Trimesh

    if TYPE_CHECKING:
        from pytorch3d.structures import Meshes
except ImportError as e:
    msg = '3D features are not available. Please install the required dependencies with: pip install "tinytools[3d]"'
    raise ImportError(msg) from e


# Initialize a logger
logger = get_logger(__name__)


def to_trimesh(meshes: Meshes) -> list[Trimesh]:
    """Convert PyTorch3D Meshes objects to Trimesh objects.

    This function handles different texture types (vertex colors, atlas colors) and correctly extracts vertex and
    face information for creating Trimesh objects.

    Args:
        meshes(Meshes): A Pytorch3D Meshes object.

    Returns:
        list[Trimesh]: A list of Trimesh objects corresponding to the input Meshes.

    Raises:
        NotImplementedError: If the texture type is TexturesUV.

    """
    trimeshes = []

    for idx, (verts, faces) in enumerate(zip(meshes.verts_list(), meshes.faces_list(), strict=True)):
        textures = None if meshes.textures is None else meshes.textures[idx]
        verts_colors = face_colors = None

        if textures is not None:
            if isinstance(textures, TexturesVertex):
                verts_colors = textures.verts_features_list()[0].detach().cpu().numpy()
            elif isinstance(textures, TexturesAtlas):
                atlas = textures.atlas_list()[0]
                if atlas.shape[-3:] == (1, 1, 3):
                    face_colors = atlas.view(-1, 3).detach().cpu().numpy()
            else:
                logger.warning("TexturesUV is not yet supported. Texture sampling will be used instead.")

            if verts_colors is None and face_colors is None:
                # source: https://github.com/facebookresearch/pytorch3d/issues/854#issuecomment-925737629
                verts_colors_packed = torch.zeros_like(verts)
                verts_colors_packed[faces] = textures.faces_verts_textures_packed()
                verts_colors = verts_colors_packed.detach().cpu().numpy()

        trimeshes.append(
            Trimesh(
                vertices=verts.detach().cpu().numpy(),
                faces=faces.detach().cpu().numpy(),
                vertex_colors=verts_colors,
                face_colors=face_colors,
                process=False,
            )
        )

    return trimeshes
