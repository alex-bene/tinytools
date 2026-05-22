"""Camera utilities supporting NumPy arrays and PyTorch tensors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import utils3d as u3d

from tinytools.array_ops import all_along, arraytensor, atan, cast_dtype, deg2rad, get_device, numel, rad2deg
from tinytools.imports import module_from_obj, optional_module

if TYPE_CHECKING:
    import torch  # pyright: ignore[reportMissingImports]

    ArrayTensor: TypeAlias = np.ndarray | torch.Tensor
else:
    torch = optional_module("torch", extra="torch")

    ArrayTensor = Any


def focal_to_fov(
    focal_length: float | ArrayTensor,
    image_size_px: int | ArrayTensor | None = None,
    *,
    is_pixels: bool = True,
    in_degrees: bool = True,
) -> float | ArrayTensor:
    """Convert focal length to field of view.

    Args:
        focal_length (float | ArrayTensor): Focal length. If batched, it must be
            broadcastable with `image_size_px`. Shape: (...)
        image_size_px (int | ArrayTensor | None, optional): Image size in pixels for
            the matching axis. If None, `is_pixels` must be False. Shape: (...)
        is_pixels (bool, optional): If True, interpret `focal_length` in pixels. Default: True.
        in_degrees (bool, optional): If True, return FOV angles in degrees. Default: True.

    Returns:
        float | ArrayTensor: Field of view. Shape: (...)

    """
    if isinstance(focal_length, (Sequence, int, float)):
        focal_length = arraytensor(focal_length, module=np, dtype=np.float32)

    module = module_from_obj(focal_length)
    device = get_device(focal_length)
    if isinstance(image_size_px, (Sequence, int, float)):
        image_size_px = arraytensor(image_size_px, module=module, dtype=module.float32, device=device)

    if is_pixels:
        if image_size_px is None:
            msg = "image_size_px is required when focal_length is in pixels."
            raise ValueError(msg)
        focal_length = focal_length / image_size_px

    focal_length = cast_dtype(focal_length, "float32")
    fov = u3d.focal_to_fov(focal_length)
    if in_degrees:
        fov = rad2deg(fov)
    return _maybe_to_python_scalar(cast_dtype(fov, "float32"))


def fov_to_focal(
    fov: float | ArrayTensor,
    image_size_px: int | ArrayTensor | None = None,
    is_degrees: bool = True,
    in_pixels: bool = True,
) -> float | ArrayTensor:
    """Convert field of view to focal length in pixels.

    Args:
        fov (float | ArrayTensor): Field of view for the matching axis. Shape: (...)
        image_size_px (int | ArrayTensor | None, optional): Image size in pixels for
            the matching axis. If None, `in_pixels` must be False. Shape: (...)
        is_degrees (bool, optional): If True, interpret `fov` in degrees. Default: True.
        in_pixels (bool, optional): If True, return focal length in pixels. Default: True.

    Returns:
        float | ArrayTensor: Focal length in pixels. Shape: (...)

    """
    if in_pixels and image_size_px is None:
        msg = "image_size_px is required when output is in pixels."
        raise ValueError(msg)

    if isinstance(fov, (Sequence, int, float)):
        fov = arraytensor(fov, module=np, dtype=np.float32)
    if isinstance(image_size_px, (Sequence, int, float)):
        image_size_px = arraytensor(image_size_px, module=np, dtype=np.float32)

    if is_degrees:
        fov = deg2rad(fov)
    focal = u3d.fov_to_focal(fov)
    if in_pixels:
        focal *= image_size_px
    return _maybe_to_python_scalar(focal)


def fov_to_intrinsics(
    fov: float | ArrayTensor,
    image_size_hw: tuple[int, int] | ArrayTensor | None = None,
    in_degrees: bool = True,
    in_pixels: bool = True,
) -> ArrayTensor:
    """Convert horizontal FOV to an OpenCV intrinsics matrix.

    Args:
        fov (float | ArrayTensor): Horizontal field of view. Shape: (...)
        image_size_hw (tuple[int, int] | ArrayTensor | None, optional): Image size
            as `(H, W)`. If batched, the trailing shape must be `(2,)`. Shape: (..., 2)
        in_degrees (bool, optional): If True, interpret `fov` in degrees. Default: True.
        in_pixels (bool, optional): If True, return intrinsics in pixels; otherwise
            return normalized intrinsics. Default: True.

    Returns:
        ArrayTensor: OpenCV intrinsics. Shape: (..., 3, 3)

    """
    if in_pixels and image_size_hw is None:
        msg = "image_size_hw is required when output is in pixels."
        raise ValueError(msg)

    if isinstance(fov, (Sequence, int, float)):
        fov = arraytensor(fov, module=np, dtype=np.float32)

    module = module_from_obj(fov)
    device = get_device(fov)

    if in_degrees:
        fov = deg2rad(fov)

    aspect_ratio: float | ArrayTensor = 1.0
    if image_size_hw is not None:
        if isinstance(image_size_hw, (Sequence, int, float)):
            image_size_hw = arraytensor(image_size_hw, module=module, dtype=module.float32, device=device)

        if image_size_hw.shape[-1] != 2:
            msg = f"image_size_hw must have trailing shape (2,), got {image_size_hw.shape}"
            raise ValueError(msg)

        image_height = image_size_hw[..., 0]
        image_width = image_size_hw[..., 1]
        aspect_ratio = image_width / image_height

    intrinsics = u3d.intrinsics_from_fov(fov_x=fov, aspect_ratio=aspect_ratio)
    if in_pixels:
        intrinsics = u3d.denormalize_intrinsics(intrinsics, size=image_size_hw)
    return intrinsics


def infer_fov_from_pointmap(pointmap: ArrayTensor) -> float | ArrayTensor:
    """Estimate horizontal FOV from a 3D pointmap in camera coordinates.

    The estimator fits robust focal candidates from many valid 2D-3D correspondences and converts the
    larger of the horizontal and vertical focal estimates into a conservative horizontal FOV.

    Args:
        pointmap (ArrayTensor): Pointmap in camera coordinates. The trailing shape must be `(H, W, 3)`.
            Shape: (..., H, W, 3)

    Returns:
        float | ArrayTensor: Estimated horizontal FOV in degrees. Shape: (...)

    Raises:
        ValueError: If `pointmap` does not end with shape `(H, W, 3)` or focal estimation fails.

    """
    module: torch = module_from_obj(pointmap)
    device = get_device(pointmap)

    pointmap_arr = arraytensor(pointmap, module=module, device=device)
    if pointmap_arr.ndim < 3 or pointmap_arr.shape[-1] != 3:
        msg = f"Expected pointmap shape (..., H, W, 3), got {pointmap_arr.shape}"
        raise ValueError(msg)

    batch_shape = pointmap_arr.shape[:-3]
    if not batch_shape:
        return _infer_fov_single(pointmap_arr)

    flat_batch = int(np.prod(batch_shape))
    flat_pointmaps = pointmap_arr.reshape(flat_batch, *pointmap_arr.shape[-3:])
    estimates = [_infer_fov_single(flat_pointmaps[idx]) for idx in range(flat_batch)]

    return module.stack(estimates).reshape(batch_shape)


def _maybe_to_python_scalar(value: float | ArrayTensor) -> float | ArrayTensor:
    """Return Python floats for scalar NumPy outputs and leave arrays/tensors unchanged."""
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return float(value)
    if isinstance(value, np.generic):
        return float(value)
    return value


def _infer_fov_single(pointmap: ArrayTensor) -> ArrayTensor:
    """Infer FOV for a single pointmap with shape `(H, W, 3)`."""
    image_height, image_width = pointmap.shape[:2]
    cx = image_width / 2.0
    cy = image_height / 2.0
    eps = 1e-8

    module = module_from_obj(pointmap)
    arrange_extra_kwargs = {"device": get_device(pointmap)} if module.__name__ == "torch" else {}
    v_coords, u_coords = module.meshgrid(
        module.arange(image_height, dtype=module.float32, **arrange_extra_kwargs),
        module.arange(image_width, dtype=module.float32, **arrange_extra_kwargs),
        indexing="ij",
    )
    finite = all_along(module.isfinite(pointmap), dim=-1)
    positive_depth = pointmap[..., 2] > eps
    valid = finite & positive_depth
    valid_count = int(valid.sum().item()) if module.__name__ == "torch" else int(valid.sum())

    if valid_count < 20:
        msg = "Not enough valid pixels for robust focal estimation."
        raise ValueError(msg)

    x_over_z = pointmap[..., 0][valid] / pointmap[..., 2][valid]
    y_over_z = pointmap[..., 1][valid] / pointmap[..., 2][valid]
    du = u_coords[valid] - cx
    dv = v_coords[valid] - cy

    x_mask = module.abs(x_over_z) > eps
    y_mask = module.abs(y_over_z) > eps
    fx_candidates = module.abs(du[x_mask] / x_over_z[x_mask])
    fy_candidates = module.abs(dv[y_mask] / y_over_z[y_mask])

    if numel(fx_candidates) == 0 or numel(fy_candidates) == 0:
        msg = "Could not estimate both fx and fy from valid pointmap pixels."
        raise ValueError(msg)

    fx_est = _robust_positive_median(fx_candidates)
    fy_est = _robust_positive_median(fy_candidates)
    focal = module.maximum(fx_est, fy_est)
    half_fov_rad = atan(
        arraytensor(image_width * 0.5, module=module, dtype=module.float32, device=get_device(pointmap)) / focal
    )
    return cast_dtype(rad2deg(half_fov_rad * 2.0), "float32")


def _robust_positive_median(values: ArrayTensor) -> ArrayTensor:
    """Return a robust median after filtering to positive finite values."""
    eps = 1e-8
    module: torch = module_from_obj(values)
    filtered = values[module.isfinite(values) & (values > eps)]
    if numel(filtered) == 0:
        msg = "No valid focal candidates remained after filtering."
        raise ValueError(msg)
    median = module.median(filtered)
    mad = module.median(module.abs(filtered - median))
    if float(mad.item()) <= eps:
        return cast_dtype(median, "float32")
    robust_sigma = mad * 1.4826
    inliers = module.abs(filtered - median) <= (3.0 * robust_sigma)
    inlier_values = filtered[inliers]
    if numel(inlier_values) == 0:
        return cast_dtype(median, "float32")
    return cast_dtype(module.median(inlier_values), "float32")
