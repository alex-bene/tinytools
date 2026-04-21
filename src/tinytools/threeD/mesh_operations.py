"""Mesh processing operations for trimesh objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tinytools import get_logger
from tinytools.imports import optional_module

if TYPE_CHECKING:
    import torch
    import trimesh
else:
    trimesh = optional_module("trimesh")
    torch = optional_module("torch")

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


def transfer_visual(
    *, mesh_source: trimesh.Trimesh, mesh_target: trimesh.Trimesh, device: torch.device | str | None = None
) -> None:
    """Attach source visual (texture / vertex colors / face colors) to a target mesh in place.

    For each target vertex (or face centroid, for face colors), the closest point on the source
    mesh is found and the visual is resampled: UVs and per-vertex colors via barycentric
    interpolation, per-face colors via the closest source triangle. Useful when a target mesh has
    been derived from the source (e.g., by simplification, remeshing, or external processing) and
    should inherit its appearance.

    Args:
        mesh_source (trimesh.Trimesh): Source mesh carrying the visual to transfer.
        mesh_target (trimesh.Trimesh): Target mesh whose ``visual`` will be updated in place.
        device (torch.device | str, optional): Device for PyTorch3D-accelerated closest point
            computation. If None, uses trimesh CPU implementation. Default: None.

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
                mesh_source=mesh_source, query_points=np.asarray(mesh_target.vertices, dtype=np.float64), device=device
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
                mesh_source=mesh_source, query_points=np.asarray(mesh_target.vertices, dtype=np.float64), device=device
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
            triangle_ids, _ = _closest_triangle_barycentric(
                mesh_source=mesh_source, query_points=centroids, device=device
            )
            if triangle_ids is not None and len(triangle_ids) == len(centroids):
                mesh_target.visual = trimesh.visual.ColorVisuals(
                    mesh=mesh_target, face_colors=source_face_colors[triangle_ids]
                )
                return


def _closest_triangle_barycentric_pt3d(
    *, triangles: np.ndarray, query_points: np.ndarray, device: torch.device | str
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """PyTorch3D-accelerated closest point on triangles with barycentric coordinates.

    Uses PyTorch3D's internal CUDA kernel for fast closest-face lookup, then computes
    barycentric coordinates on the closest faces. The kernel only accepts float32 inputs; on
    meshes with very small triangles it can return far-away faces with near-zero distances unless
    the geometry is normalized first. Centering and scaling are therefore part of the algorithm,
    not cosmetic preprocessing.

    Args:
        triangles (np.ndarray): Triangle vertices. Shape: (F, 3, 3).
        query_points (np.ndarray): Points to project. Shape: (N, 3).
        device (torch.device | str): Device for computation.

    Returns:
        tuple[np.ndarray | None, np.ndarray | None]: (triangle_ids, barycentric_weights).
            Shapes: (N,) and (N, 3).

    """
    from pytorch3d import _C  # noqa: PLC0415

    triangles_np = np.asarray(triangles, dtype=np.float64)
    query_points_np = np.asarray(query_points, dtype=np.float64)

    center = triangles_np.reshape(-1, 3).mean(axis=0)
    edge_01 = triangles_np[:, 1] - triangles_np[:, 0]
    edge_02 = triangles_np[:, 2] - triangles_np[:, 0]
    triangle_areas = 0.5 * np.linalg.norm(np.cross(edge_01, edge_02), axis=1)
    positive_areas = triangle_areas[triangle_areas > 0]
    scale = 1.0 / max(float(np.sqrt(np.median(positive_areas))), 1e-12) if len(positive_areas) > 0 else 1.0

    normalized_triangles = (triangles_np - center) * scale
    normalized_query_points = (query_points_np - center) * scale

    # Keep this normalization even though closest-point queries are scale/translation invariant.
    # The PyTorch3D C kernel is float32-only and was observed to choose far-away triangles on
    # tiny-triangle meshes when called in the original coordinate scale.
    tris_f32 = torch.as_tensor(normalized_triangles, dtype=torch.float32, device=device)
    pts_f32 = torch.as_tensor(normalized_query_points, dtype=torch.float32, device=device)

    n_pts = pts_f32.shape[0]

    # Use PyTorch3D's CUDA kernel for fast closest face lookup
    pts_first_idx = torch.tensor([0], dtype=torch.int64, device=device)
    tris_first_idx = torch.tensor([0], dtype=torch.int64, device=device)

    min_triangle_area = 1e-8 * scale * scale
    _, triangle_ids = _C.point_face_dist_forward(
        pts_f32, pts_first_idx, tris_f32, tris_first_idx, n_pts, min_triangle_area
    )

    # Convert to float64 for higher precision in barycentric computation
    tris = tris_f32.to(torch.float64)
    pts = pts_f32.to(torch.float64)

    # Gather the closest triangles for each query point
    closest_tris = tris[triangle_ids]  # (N, 3, 3)

    # Two-step approach (mirrors trimesh):
    # 1. Compute actual closest points on the triangles
    closest_pts = _closest_point_on_triangles(pts, closest_tris)
    # 2. Compute barycentric coordinates for points that are ON the triangles
    barycentric = _points_to_barycentric(closest_pts, closest_tris)

    triangle_ids_np = triangle_ids.cpu().numpy().astype(np.int64)
    barycentric_np = barycentric.cpu().numpy().astype(np.float64)

    if not np.isfinite(barycentric_np).all():
        return None, None
    return triangle_ids_np, barycentric_np


def _closest_point_on_edge(points: torch.Tensor, edge_start: torch.Tensor, edge_end: torch.Tensor) -> torch.Tensor:
    """Compute closest point on edge segment for each query point.

    Args:
        points (torch.Tensor): Query points. Shape: (N, 3).
        edge_start (torch.Tensor): Edge start vertices. Shape: (N, 3).
        edge_end (torch.Tensor): Edge end vertices. Shape: (N, 3).

    Returns:
        torch.Tensor: Closest points on edges. Shape: (N, 3).

    """
    edge = edge_end - edge_start
    edge_len_sq = (edge * edge).sum(dim=-1, keepdim=True).clamp(min=1e-12)
    t = ((points - edge_start) * edge).sum(dim=-1, keepdim=True) / edge_len_sq
    t = t.clamp(0, 1)
    return edge_start + t * edge


def _closest_point_on_triangles(points: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    """Compute the closest point on each triangle to the corresponding query point.

    Args:
        points (torch.Tensor): Query points. Shape: (N, 3).
        triangles (torch.Tensor): Triangle vertices, one per point. Shape: (N, 3, 3).

    Returns:
        torch.Tensor: Closest points on triangles. Shape: (N, 3).

    """
    v0 = triangles[:, 0]
    v1 = triangles[:, 1]
    v2 = triangles[:, 2]

    # Project point onto triangle plane and get barycentric coords
    e0 = v1 - v0
    e1 = v2 - v0
    d = points - v0

    d00 = (e0 * e0).sum(dim=-1)
    d01 = (e0 * e1).sum(dim=-1)
    d11 = (e1 * e1).sum(dim=-1)
    d20 = (d * e0).sum(dim=-1)
    d21 = (d * e1).sum(dim=-1)

    denom = (d00 * d11 - d01 * d01).clamp(min=1e-12)
    v = (d11 * d20 - d01 * d21) / denom  # barycentric coord for v1
    w = (d00 * d21 - d01 * d20) / denom  # barycentric coord for v2
    u = 1 - v - w  # barycentric coord for v0

    # Project onto plane: this is the closest point IF inside triangle
    normal = torch.cross(e0, e1, dim=-1)
    normal_len_sq = (normal * normal).sum(dim=-1, keepdim=True).clamp(min=1e-12)
    normal_unit = normal / normal_len_sq.sqrt()
    dist_to_plane = (d * normal_unit).sum(dim=-1, keepdim=True)
    plane_proj = points - dist_to_plane * normal_unit

    # Check if inside triangle (all barycentric coords >= 0)
    inside = (u >= 0) & (v >= 0) & (w >= 0)

    # For points outside, compute closest point on each edge and pick nearest
    closest_e01 = _closest_point_on_edge(points, v0, v1)
    closest_e12 = _closest_point_on_edge(points, v1, v2)
    closest_e02 = _closest_point_on_edge(points, v0, v2)

    dist_e01 = ((points - closest_e01) ** 2).sum(dim=-1)
    dist_e12 = ((points - closest_e12) ** 2).sum(dim=-1)
    dist_e02 = ((points - closest_e02) ** 2).sum(dim=-1)

    # Find which edge is closest
    _, min_idx = torch.stack([dist_e01, dist_e12, dist_e02], dim=-1).min(dim=-1)
    edge_closest = torch.stack([closest_e01, closest_e12, closest_e02], dim=1)
    closest_on_edge = edge_closest[torch.arange(len(points), device=points.device), min_idx]

    # Use plane projection if inside, edge projection if outside
    return torch.where(inside.unsqueeze(-1), plane_proj, closest_on_edge)


def _points_to_barycentric(points: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    """Compute barycentric coordinates for points on triangles.

    Assumes points are already on the triangle plane.

    Args:
        points (torch.Tensor): Points on triangles. Shape: (N, 3).
        triangles (torch.Tensor): Triangle vertices. Shape: (N, 3, 3).

    Returns:
        torch.Tensor: Barycentric coordinates (w0, w1, w2). Shape: (N, 3).

    """
    v0 = triangles[:, 0]
    v1 = triangles[:, 1]
    v2 = triangles[:, 2]

    e0 = v1 - v0
    e1 = v2 - v0
    e2 = points - v0

    d00 = (e0 * e0).sum(dim=-1)
    d01 = (e0 * e1).sum(dim=-1)
    d11 = (e1 * e1).sum(dim=-1)
    d20 = (e2 * e0).sum(dim=-1)
    d21 = (e2 * e1).sum(dim=-1)

    denom = (d00 * d11 - d01 * d01).clamp(min=1e-12)
    w1 = (d11 * d20 - d01 * d21) / denom
    w2 = (d00 * d21 - d01 * d20) / denom
    w0 = 1 - w1 - w2

    return torch.stack([w0, w1, w2], dim=-1)


def _closest_triangle_barycentric(
    *, mesh_source: trimesh.Trimesh, query_points: np.ndarray, device: torch.device | str | None = None
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Project query points onto the source mesh and return barycentric weights on the hit triangle.

    Args:
        mesh_source (trimesh.Trimesh): Source mesh.
        query_points (np.ndarray): Points to project. Shape: (N, 3).
        device (torch.device | str, optional): Device for PyTorch3D-accelerated computation.
            If None, uses trimesh CPU implementation. Default: None.

    Returns:
        tuple[np.ndarray | None, np.ndarray | None]: (triangle_ids, barycentric_weights). Returns
            ``(None, None)`` when the projection or barycentric computation produces non-finite values.
            Shapes: (N,) and (N, 3).

    """
    if device is not None and torch.device(device).type != "cpu":
        return _closest_triangle_barycentric_pt3d(
            triangles=mesh_source.triangles, query_points=query_points, device=device
        )

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
