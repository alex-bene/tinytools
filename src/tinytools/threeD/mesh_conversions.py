"""Meshes class for TinyHumans.

This module defines the Meshes and BodyMeshes classes, which extend PyTorch3D's Meshes class to provide additional
functionality for working with 3D meshes, including conversion to Trimesh objects and handling of body-specific
parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from PIL import Image

from tinytools import get_logger
from tinytools.imports import optional_attr, optional_module

from .coordinate_conversions import CoordinateConversions

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch  # pyright: ignore[reportMissingImports]
    from pytorch3d.renderer import TexturesAtlas, TexturesUV, TexturesVertex  # pyright: ignore[reportMissingImports]
    from pytorch3d.structures import Meshes  # pyright: ignore[reportMissingImports]
    from trimesh import Trimesh  # pyright: ignore[reportMissingImports]
    from trimesh.visual.texture import TextureVisuals  # pyright: ignore[reportMissingImports]
else:
    torch = optional_module("torch")
    Meshes = optional_attr("pytorch3d.structures", "Meshes", package="pytorch3d")
    TexturesAtlas = optional_attr("pytorch3d.renderer", "TexturesAtlas", package="pytorch3d")
    TexturesUV = optional_attr("pytorch3d.renderer", "TexturesUV", package="pytorch3d")
    TexturesVertex = optional_attr("pytorch3d.renderer", "TexturesVertex", package="pytorch3d")
    TextureVisuals = optional_attr("trimesh.visual.texture", "TextureVisuals", package="trimesh")
    Trimesh = optional_attr("trimesh", "Trimesh", package="trimesh")

CoordinateSystem = Literal["opencv", "pytorch3d"]

# Initialize a logger
logger = get_logger(__name__)


def pt3d_to_trimesh(meshes: Meshes) -> list[Trimesh]:
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


def trimesh_to_pt3d(
    meshes: Trimesh | Sequence[Trimesh],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
    source_coords: CoordinateSystem = "opencv",
) -> Meshes:
    """Convert trimesh objects to PyTorch3D Meshes.

    Handles visual transfer from trimesh to PyTorch3D textures:

    - ``TextureVisuals`` with UVs + material image: creates ``TexturesUV`` with the material image
      as a texture map.
    - ``ColorVisuals`` with per-vertex colors: creates ``TexturesVertex``.
    - ``ColorVisuals`` with per-face colors: creates ``TexturesAtlas`` with 1x1 atlas per face.

    For mixed batches where some meshes have colors and others don't, white placeholders are used.

    Args:
        meshes (trimesh.Trimesh | Sequence[trimesh.Trimesh]): Single trimesh or sequence of trimeshes.
        device (torch.device | str, optional): Target device for tensors. Default: "cpu".
        dtype (torch.dtype, optional): Data type for vertex tensors. Default: torch.float32.
        source_coords (CoordinateSystem, optional): Coordinate system of input meshes. If "opencv",
            vertices are transformed to PyTorch3D coordinates (X-right, Y-down, Z-forward to
            X-left, Y-up, Z-forward). If "pytorch3d", no transformation is applied. Default: "opencv".

    Returns:
        Meshes: PyTorch3D Meshes object with textures transferred from the input trimeshes.

    """
    if dtype is None:
        dtype = torch.float32

    if not isinstance(meshes, (list, tuple)):
        meshes = [meshes]

    verts_list: list[torch.Tensor] = []
    faces_list: list[torch.Tensor] = []
    uv_textures: list[TexturesUV] = []
    verts_colors_list: list[torch.Tensor] = []
    face_colors_list: list[torch.Tensor] = []
    has_uv = False
    has_vertex_colors = False
    has_face_colors = False

    coord_transform = (
        torch.as_tensor(CoordinateConversions.cv_to_pt3d, device=device, dtype=dtype)
        if source_coords == "opencv"
        else None
    )

    for mesh in meshes:
        verts = torch.as_tensor(np.asarray(mesh.vertices), device=device, dtype=dtype)
        if coord_transform is not None:
            verts = verts @ coord_transform
        verts_list.append(verts)

        faces = torch.as_tensor(np.asarray(mesh.faces), device=device, dtype=torch.long)
        faces_list.append(faces)

        uv_tex, vert_colors, face_colors = _extract_trimesh_colors(mesh, device=device, dtype=dtype)

        has_uv = has_uv or uv_tex is not None
        uv_textures.append(uv_tex)  # type: ignore[arg-type]

        has_vertex_colors = has_vertex_colors or vert_colors is not None
        verts_colors_list.append(vert_colors if vert_colors is not None else torch.ones_like(verts))

        has_face_colors = has_face_colors or face_colors is not None
        face_colors_list.append(
            face_colors if face_colors is not None else torch.ones((len(faces), 1, 1, 3), device=device, dtype=dtype)
        )

    textures = _build_batch_textures(
        uv_textures, verts_colors_list, face_colors_list, has_uv, has_vertex_colors, has_face_colors
    )

    return Meshes(verts=verts_list, faces=faces_list, textures=textures)


def _extract_trimesh_colors(
    mesh: Trimesh, *, device: torch.device | str, dtype: torch.dtype
) -> tuple[TexturesUV | None, torch.Tensor | None, torch.Tensor | None]:
    """Extract UV texture, vertex colors, and face colors from trimesh visual.

    Args:
        mesh (trimesh.Trimesh): Input trimesh.
        device (torch.device | str): Target device.
        dtype (torch.dtype): Data type for color tensors.

    Returns:
        tuple[TexturesUV | None, Tensor | None, Tensor | None]:
            - ``uv_texture`` (TexturesUV | None): UV texture if available.
            - ``vertex_colors`` (Tensor | None): Per-vertex RGB colors in [0, 1]. Shape: (V, 3).
            - ``face_colors`` (Tensor | None): Per-face RGB colors as atlas. Shape: (F, 1, 1, 3).

    """
    visual = getattr(mesh, "visual", None)
    if visual is None:
        return None, None, None

    uv_texture = _try_texture_uv(visual, device, dtype)
    if uv_texture is not None:
        return uv_texture, None, None

    visual_kind = getattr(visual, "kind", None)
    vertex_colors: torch.Tensor | None = None
    face_colors: torch.Tensor | None = None

    if visual_kind == "vertex" or hasattr(visual, "vertex_colors"):
        vert_cols = getattr(visual, "vertex_colors", None)
        if vert_cols is not None and len(vert_cols) == len(mesh.vertices):
            vert_cols_np = np.asarray(vert_cols)
            if vert_cols_np.shape[-1] >= 3:
                vert_cols_rgb = vert_cols_np[..., :3].astype(np.float32)
                if vert_cols_rgb.max() > 1.0:
                    vert_cols_rgb = vert_cols_rgb / 255.0
                vertex_colors = torch.as_tensor(vert_cols_rgb, device=device, dtype=dtype)

    if vertex_colors is None and (visual_kind == "face" or hasattr(visual, "face_colors")):
        face_cols = getattr(visual, "face_colors", None)
        if face_cols is not None and len(face_cols) == len(mesh.faces):
            face_cols_np = np.asarray(face_cols)
            if face_cols_np.shape[-1] >= 3:
                face_cols_rgb = face_cols_np[..., :3].astype(np.float32)
                if face_cols_rgb.max() > 1.0:
                    face_cols_rgb = face_cols_rgb / 255.0
                face_colors = torch.as_tensor(face_cols_rgb, device=device, dtype=dtype).view(-1, 1, 1, 3)

    return None, vertex_colors, face_colors


def _try_texture_uv(visual: object, device: torch.device | str, dtype: torch.dtype) -> TexturesUV | None:
    """Try to extract TexturesUV from a TextureVisuals object.

    Args:
        visual: Trimesh TextureVisuals object.
        device (torch.device | str): Target device.
        dtype (torch.dtype): Data type for float tensors.

    Returns:
        TexturesUV | None: Texture or None if extraction fails.

    """
    if not isinstance(visual, TextureVisuals):
        return None

    uv = getattr(visual, "uv", None)
    material = getattr(visual, "material", None)
    image = getattr(material, "image", None) if material is not None else None
    if uv is None or image is None:
        return None

    uv_array = np.asarray(uv, dtype=np.float64)
    if uv_array.ndim != 2 or uv_array.shape[1] != 2:
        return None

    try:
        if not isinstance(image, Image.Image):
            return None
        img_array = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
        texture_map = torch.as_tensor(img_array, device=device, dtype=dtype).unsqueeze(0)
        uv_tensor = torch.as_tensor(uv_array, device=device, dtype=dtype).unsqueeze(0)
        return TexturesUV(maps=texture_map, faces_uvs=None, verts_uvs=uv_tensor)
    except AttributeError:
        return None


def _build_batch_textures(
    uv_textures: list[TexturesUV | None],
    verts_colors_list: list[torch.Tensor],
    face_colors_list: list[torch.Tensor],
    has_uv: bool,
    has_vertex_colors: bool,
    has_face_colors: bool,
) -> TexturesUV | TexturesVertex | TexturesAtlas | None:
    """Build batched textures from extracted color data.

    Args:
        uv_textures: List of UV textures (may contain None).
        verts_colors_list: List of vertex color tensors (with white placeholders).
        face_colors_list: List of face color tensors (with white placeholders).
        has_uv: Whether any mesh has UV textures.
        has_vertex_colors: Whether any mesh has vertex colors.
        has_face_colors: Whether any mesh has face colors.

    Returns:
        TexturesUV | TexturesVertex | TexturesAtlas | None: Batched texture or None.

    """
    all_have_uv = has_uv and all(t is not None for t in uv_textures)

    if all_have_uv:
        if has_vertex_colors or has_face_colors:
            logger.warning(
                "Batch contains UV textures alongside vertex/face colors; "
                "only UV textures will be used, other colors will be discarded."
            )
        return TexturesUV(
            maps=[t.maps_list()[0] for t in uv_textures],  # type: ignore[union-attr]
            verts_uvs=[t.verts_uvs_list()[0] for t in uv_textures],  # type: ignore[union-attr]
            faces_uvs=[
                t.faces_uvs_list()[0] if t.faces_uvs_list()[0] is not None else None
                for t in uv_textures  # type: ignore[union-attr]
            ],
        )

    if has_uv:
        logger.warning(
            "Some meshes have UV textures but not all; UV textures will be discarded, "
            "falling back to vertex/face colors or white placeholders."
        )

    if has_vertex_colors:
        if has_face_colors:
            logger.warning(
                "Batch contains both vertex colors and face colors; "
                "only vertex colors will be used, face colors will be discarded."
            )
        return TexturesVertex(verts_features=verts_colors_list)

    if has_face_colors:
        return TexturesAtlas(atlas=face_colors_list)

    return None
