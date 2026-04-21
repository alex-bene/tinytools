"""Mesh processing operations for trimesh objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tinytools import get_logger
from tinytools.imports import optional_module

if TYPE_CHECKING:
    import trimesh
else:
    trimesh = optional_module("trimesh")

logger = get_logger(__name__)


def simplify_mesh(  # noqa: PLR0911
    mesh: trimesh.Trimesh,
    *,
    percent: float | None = None,
    face_count: int | None = None,
    aggression: int | None = None,
    clean_topology: bool = False,
    fill_holes: bool = False,
    preserving_visual: bool = True,
) -> trimesh.Trimesh:
    """Simplify a trimesh via quadric decimation while, optionally, preserving its visual representation.

    Forwards ``percent``, ``face_count``, and ``aggression`` to
    ``trimesh.Trimesh.simplify_quadric_decimation`` and then transfers the source visual to the
    simplified mesh in the most faithful form it supports:

    - ``TextureVisuals`` with UVs + a material image: resamples per-vertex UVs on the simplified
      mesh (via barycentric interpolation against the closest source triangle) and reuses the
      source material. The texture image itself is preserved at full resolution.
    - ``ColorVisuals`` with per-vertex colors: barycentrically interpolates vertex colors.
    - ``ColorVisuals`` with per-face colors only: assigns each simplified face the color of the
      closest source triangle.
    - Anything else: returns the simplified mesh with no visual.

    Simplification is skipped (the input is returned unchanged) when all three target arguments
    are ``None`` or when ``face_count`` is non-positive / already >= the source face count. On
    any decimation or transfer failure, the original mesh is returned.

    Args:
        mesh (trimesh.Trimesh): Input mesh to simplify. Shape: (V, 3) vertices, (F, 3) faces.
        percent (float, optional): Target fraction of faces to retain, in (0, 1]. Forwarded to
            trimesh. Default: None.
        face_count (int, optional): Target maximum face count. Forwarded to trimesh.
            Default: None.
        aggression (int, optional): Decimation aggression level forwarded to trimesh. Higher
            values simplify more aggressively at the cost of shape fidelity. Default: None.
        clean_topology (bool, optional): If True, clean up topology artifacts left by decimation
            before transferring the visual. Operations run in this order: (1) ``merge_vertices``
            stitches vertices at (nearly) identical positions so seams/cracks become shared edges;
            (2) ``update_faces(unique_faces())`` drops duplicate faces (identical index triples,
            possibly flipped) that merging can produce and that cause z-fighting / double-counted
            area; (3) ``update_faces(nondegenerate_faces())`` drops zero-area faces (repeated
            indices or collinear vertices) that would give NaN normals; (4)
            ``remove_unreferenced_vertices`` removes vertices no remaining face points at.
            Default: False.
        fill_holes (bool, optional): If True, fill small boundary holes in the simplified mesh via
            ``trimesh.Trimesh.fill_holes``. Runs after ``clean_topology`` so seams are stitched
            before boundary loops are detected. Default: False.
        preserving_visual (bool, optional): If True, transfer the source visual to the simplified
            mesh. Default: True.

    Returns:
        trimesh.Trimesh: Simplified mesh with preserved visual, or the original mesh on fallback.

    """
    if percent is None and face_count is None and aggression is None:
        return mesh

    if face_count is not None:
        face_count = int(face_count)
        if face_count <= 0 or len(mesh.faces) <= face_count:
            return mesh

    try:
        simplified = mesh.simplify_quadric_decimation(percent=percent, face_count=face_count, aggression=aggression)
    except (ImportError, ModuleNotFoundError, RuntimeError, TypeError, ValueError):
        logger.warning("Mesh simplification failed; falling back to original mesh.", exc_info=True)
        return mesh

    if len(simplified.vertices) == 0 or len(simplified.faces) == 0:
        logger.warning("Simplified mesh is empty; falling back to original mesh.")
        return mesh

    if clean_topology:
        simplified.merge_vertices()
        simplified.update_faces(simplified.unique_faces())
        simplified.update_faces(simplified.nondegenerate_faces())
        simplified.remove_unreferenced_vertices()

    if fill_holes:
        simplified.fill_holes()

    if len(simplified.vertices) == 0 or len(simplified.faces) == 0:
        logger.warning("Simplified mesh is empty after cleanup; falling back to original mesh.")
        return mesh

    if not preserving_visual:
        return simplified

    try:
        transfer_visual(mesh_source=mesh, mesh_target=simplified)
    except (RuntimeError, TypeError, ValueError):
        logger.warning(
            "Visual transfer to simplified mesh failed; returning simplified mesh without visual.", exc_info=True
        )
    return simplified


def transfer_visual(*, mesh_source: trimesh.Trimesh, mesh_target: trimesh.Trimesh) -> None:
    """Attach source visual (texture / vertex colors / face colors) to a target mesh in place.

    For each target vertex (or face centroid, for face colors), the closest point on the source
    mesh is found and the visual is resampled: UVs and per-vertex colors via barycentric
    interpolation, per-face colors via the closest source triangle. Useful when a target mesh has
    been derived from the source (e.g., by simplification, remeshing, or external processing) and
    should inherit its appearance.

    Args:
        mesh_source (trimesh.Trimesh): Source mesh carrying the visual to transfer.
        mesh_target (trimesh.Trimesh): Target mesh whose ``visual`` will be updated in place.

    """
    source_visual = getattr(mesh_source, "visual", None)
    if source_visual is None:
        return

    if isinstance(source_visual, trimesh.visual.texture.TextureVisuals):
        source_uv = np.asarray(source_visual.uv) if source_visual.uv is not None else None
        if (
            source_uv is not None
            and source_uv.ndim == 2
            and source_uv.shape == (len(mesh_source.vertices), 2)
            and source_visual.material is not None
        ):
            triangle_ids, barycentric = _closest_triangle_barycentric(
                mesh_source=mesh_source, query_points=np.asarray(mesh_target.vertices, dtype=np.float64)
            )
            if triangle_ids is not None and barycentric is not None:
                source_faces = np.asarray(mesh_source.faces, dtype=np.int64)
                triangle_uv = source_uv[source_faces[triangle_ids]].astype(np.float64)
                new_uv = np.einsum("ni,nic->nc", barycentric, triangle_uv)
                mesh_target.visual = trimesh.visual.texture.TextureVisuals(
                    uv=new_uv.astype(source_uv.dtype, copy=False), material=source_visual.material
                )
                return

    visual_kind = getattr(source_visual, "kind", None)

    if visual_kind == "vertex":
        source_colors = _as_rgba_uint8(np.asarray(source_visual.vertex_colors), expected_rows=len(mesh_source.vertices))
        if source_colors is not None:
            triangle_ids, barycentric = _closest_triangle_barycentric(
                mesh_source=mesh_source, query_points=np.asarray(mesh_target.vertices, dtype=np.float64)
            )
            if triangle_ids is not None and barycentric is not None:
                source_faces = np.asarray(mesh_source.faces, dtype=np.int64)
                triangle_colors = source_colors[source_faces[triangle_ids]].astype(np.float64)
                new_colors = np.einsum("ni,nic->nc", barycentric, triangle_colors)
                new_colors = np.clip(np.round(new_colors), 0, 255).astype(np.uint8)
                mesh_target.visual = trimesh.visual.ColorVisuals(mesh=mesh_target, vertex_colors=new_colors)
                return

    if visual_kind == "face":
        source_face_colors = _as_rgba_uint8(np.asarray(source_visual.face_colors), expected_rows=len(mesh_source.faces))
        if source_face_colors is not None:
            centroids = np.asarray(mesh_target.triangles_center, dtype=np.float64)
            _, _, triangle_ids = trimesh.proximity.closest_point(mesh_source, centroids)
            if triangle_ids is not None and len(triangle_ids) == len(centroids):
                triangle_ids = np.asarray(triangle_ids, dtype=np.int64)
                mesh_target.visual = trimesh.visual.ColorVisuals(
                    mesh=mesh_target, face_colors=source_face_colors[triangle_ids]
                )
                return


def _closest_triangle_barycentric(
    *, mesh_source: trimesh.Trimesh, query_points: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Project query points onto the source mesh and return barycentric weights on the hit triangle.

    Args:
        mesh_source (trimesh.Trimesh): Source mesh.
        query_points (np.ndarray): Points to project. Shape: (N, 3).

    Returns:
        tuple[np.ndarray | None, np.ndarray | None]: (triangle_ids, barycentric_weights). Returns
            ``(None, None)`` when the projection or barycentric computation produces non-finite
            values. Shapes: (N,) and (N, 3).

    """
    closest_points, _, triangle_ids = trimesh.proximity.closest_point(mesh_source, query_points)
    if not np.isfinite(closest_points).all():
        return None, None
    triangle_ids = np.asarray(triangle_ids, dtype=np.int64)
    source_triangles = np.asarray(mesh_source.triangles, dtype=np.float64)[triangle_ids]
    barycentric = trimesh.triangles.points_to_barycentric(source_triangles, closest_points)
    if not np.isfinite(barycentric).all():
        return None, None
    return triangle_ids, barycentric


def _as_rgba_uint8(colors: np.ndarray, *, expected_rows: int) -> np.ndarray | None:
    """Normalize a color array to RGBA uint8 with the expected row count.

    Args:
        colors (np.ndarray): Candidate colors. Shape: (N, 3) or (N, 4).
        expected_rows (int): Expected number of rows.

    Returns:
        np.ndarray | None: RGBA colors with dtype uint8. Shape: (N, 4).

    """
    if colors.ndim != 2 or colors.shape[0] != expected_rows or colors.shape[1] not in {3, 4}:
        return None

    rgba = colors[:, :4].astype(np.float64, copy=False)
    if not np.isfinite(rgba).all():
        return None

    should_scale = (np.issubdtype(colors.dtype, np.floating) or np.issubdtype(colors.dtype, np.bool_)) and rgba.max(
        initial=0.0
    ) <= 1.0

    if rgba.shape[1] == 3:
        alpha = np.full((expected_rows, 1), 255.0, dtype=np.float64)
        rgba = np.concatenate([rgba, alpha], axis=1)
    if should_scale:
        rgba *= 255.0
        rgba[:, 3] = 255.0
    return np.clip(np.round(rgba), 0, 255).astype(np.uint8)
