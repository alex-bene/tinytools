"""Camera intrinsics conversion helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
import utils3d as u3d

from tinytools.imports import optional_module
from tinytools.threeD.camera import infer_fov_from_pointmap
from tinytools.torch.utils import as_float_tensor
from tinytools.validate import validate_ndim, validate_shape

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch  # pyright: ignore[reportMissingImports]

    from tinytools.array_ops import ArrayTensor
else:
    torch = optional_module("torch", extra="torch")

ImageSize = tuple[int, int] | list[int] | np.ndarray | torch.Tensor
T = TypeVar("T")


def prepare_intrinsics(
    intrinsics: Sequence[ArrayTensor | None] | ArrayTensor | None = None,
    fov_deg: Sequence[float | None] | ArrayTensor | float | None = None,
    pointmap: Sequence[ArrayTensor | None] | ArrayTensor | None = None,
    image_size_hw: Sequence[ImageSize | None] | ImageSize | None = None,
    return_type: Literal["fov_deg", "intrinsics_px", "intrinsics_norm"] = "intrinsics_px",
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Prepare camera intrinsics or FOV from partially provided inputs.

    Args:
        intrinsics (Sequence[ndarray | Tensor | None] | ndarray | Tensor | None, optional): Camera intrinsics in
            pixel or normalized units. Shape: (3, 3) or [(3, 3), ...]. Default: None.
        fov_deg (Sequence[float | None] | np.ndarray | torch.Tensor | float | None, optional): Horizontal field of view
            in degrees. Shape: () or (...,). Default: None.
        pointmap (Sequence[ndarray | Tensor | None] | ndarray | Tensor | None, optional): Pointmap used for
            image-size extraction and FOV inference. Shape: (H, W, 3) or [(H, W, 3), ...]. Default: None.
        image_size_hw (Sequence[ImageSize | None] | ImageSize | None, optional): Image size as `(H, W)`. If
            provided, it is preferred over `pointmap` for size-dependent intrinsics conversions. Default: None.
        return_type (Literal["fov_deg", "intrinsics_px", "intrinsics_norm"], optional): Desired output format.
            Default: "intrinsics_px".
        device (torch.device | None, optional): Device used when returning intrinsics tensors. If None, tensors are
            created on the default device. Default: None.

    Returns:
        torch.Tensor: Prepared output batched along the leading dimension. Shape: (B, 3, 3) for intrinsics outputs
        or (B,) for `return_type="fov_deg"`.

    """
    if intrinsics is None and fov_deg is None and pointmap is None:
        msg = "At least one of 'intrinsics', 'fov_deg', or 'pointmap' must be provided."
        raise ValueError(msg)
    if return_type not in ("fov_deg", "intrinsics_px", "intrinsics_norm"):
        msg = f"Unsupported return_type '{return_type}'."
        raise ValueError(msg)

    sequence_inputs = [
        value
        for value, single_ndim in ((intrinsics, 2), (fov_deg, 0), (pointmap, 3), (image_size_hw, 1))
        if _is_batch_container(value, single_ndim=single_ndim)
    ]
    batch_size = len(sequence_inputs[0]) if sequence_inputs else 1
    intrinsics_seq = _expand_to_batch(intrinsics, batch_size=batch_size, single_ndim=2)
    fov_seq = _expand_to_batch(fov_deg, batch_size=batch_size, single_ndim=0)
    pointmap_seq = _expand_to_batch(pointmap, batch_size=batch_size, single_ndim=3)
    image_size_hw_seq = _expand_to_batch(image_size_hw, batch_size=batch_size, single_ndim=1)
    return torch.stack(
        [
            _prepare_single_intrinsics(
                intrinsics=intrinsics_i,
                fov_deg=fov_i,
                pointmap=pointmap_i,
                image_size_hw=image_size_hw_i,
                return_type=return_type,
                device=device,
            )
            for intrinsics_i, fov_i, pointmap_i, image_size_hw_i in zip(
                intrinsics_seq, fov_seq, pointmap_seq, image_size_hw_seq, strict=True
            )
        ]
    )


def _prepare_single_intrinsics(
    *,
    intrinsics: ArrayTensor | None,
    fov_deg: float | None,
    pointmap: ArrayTensor | None,
    image_size_hw: ImageSize | None,
    return_type: Literal["fov_deg", "intrinsics_px", "intrinsics_norm"],
    device: torch.device | None,
) -> torch.Tensor:
    """Prepare camera intrinsics or FOV for one sample."""
    if intrinsics is None and fov_deg is None and pointmap is None:
        msg = "At least one of 'intrinsics', 'fov_deg', or 'pointmap' must be provided."
        raise ValueError(msg)

    if fov_deg is not None:
        fov_deg = as_float_tensor(fov_deg, device=device)
        validate_ndim(fov_deg, ndim=0, arg_name="fov_deg")

    if intrinsics is not None:
        intrinsics = as_float_tensor(intrinsics, device=device)
        validate_shape(intrinsics, shape=[3, 3], arg_name="intrinsics")
    elif fov_deg is None:
        validate_ndim(pointmap, ndim=3, arg_name="pointmap")
        validate_shape(pointmap, shape=[..., 3], arg_name="pointmap")
        fov_deg = as_float_tensor(infer_fov_from_pointmap(pointmap), device=device)

    resolved_image_size = _resolve_image_size(image_size_hw, pointmap)

    if return_type == "fov_deg" and fov_deg is not None:
        return fov_deg

    if intrinsics is None:
        intrinsics = u3d.intrinsics_from_fov(
            fov_x=torch.deg2rad(fov_deg), aspect_ratio=resolved_image_size[1] / resolved_image_size[0]
        )

    is_normalized = _intrinsics_are_normalized(intrinsics)
    if (return_type == "intrinsics_norm" and is_normalized) or (return_type == "intrinsics_px" and not is_normalized):
        return intrinsics

    if return_type in ("intrinsics_norm", "fov_deg") and not is_normalized:
        intrinsics = u3d.normalize_intrinsics(intrinsics, size=resolved_image_size)

    if return_type == "intrinsics_norm":
        return intrinsics
    if return_type == "fov_deg":
        return torch.rad2deg(u3d.intrinsics_to_fov(intrinsics)[0])
    return u3d.denormalize_intrinsics(intrinsics, size=resolved_image_size)


def _intrinsics_are_normalized(intrinsics: torch.Tensor) -> bool:
    """Return whether the intrinsics matrix should be treated as normalized."""
    cx = float(intrinsics[0, 2].item())
    cy = float(intrinsics[1, 2].item())
    return 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0


def _resolve_image_size(image_size_hw: ImageSize | None, pointmap: ArrayTensor | None) -> tuple[int, int] | None:
    """Resolve image size as (H, W)."""
    if image_size_hw is not None:
        if torch.is_tensor(image_size_hw):
            return int(image_size_hw[0].item()), int(image_size_hw[1].item())
        return int(image_size_hw[0]), int(image_size_hw[1])
    if pointmap is None:
        return None
    return int(pointmap.shape[0]), int(pointmap.shape[1])


def _is_batch_container(value: object, single_ndim: int) -> bool:
    """Return whether value is a top-level container of per-sample inputs."""
    if (torch.is_tensor(value) and value.ndim == (single_ndim + 1)) or (
        isinstance(value, np.ndarray) and value.ndim == (single_ndim + 1)
    ):
        return True
    if isinstance(value, (list, tuple)):
        if not value:
            return False
        item = value[0]
        if item is None:
            return True  # has None item so it's a batch container
        if torch.as_tensor(item).ndim == single_ndim:
            return True

    return False


def _expand_to_batch(value: Sequence[T | None] | T | None, *, batch_size: int, single_ndim: int) -> list[T | None]:
    """Expand scalar or None input to a batch-aligned list."""
    if _is_batch_container(value, single_ndim=single_ndim):
        return list(value)
    return [value] * batch_size
