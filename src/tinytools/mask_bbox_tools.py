"""Mask and bounding box tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from PIL import Image

from .array_ops import any_along, cast_dtype, max_along, min_along, stack_along
from .imports import module_from_obj

if TYPE_CHECKING:
    from torch import Tensor  # pyright: ignore[reportMissingImports]

    ArrayTensor: TypeAlias = np.ndarray | Tensor


def bboxes_center(bboxes: ArrayTensor) -> ArrayTensor:
    """Get the center of bounding boxes from (x_min, y_min, x_max, y_max)."""
    if bboxes.shape[-1] != 4:
        msg = "bboxes must be a tensor/array with shape (..., 4)"
        raise ValueError(msg)
    return (bboxes[..., :2] + bboxes[..., 2:]) / 2


def pad_bboxes(
    bboxes: list | ArrayTensor, image_sizes: list | ArrayTensor | None, padding_perc: float = 0.1
) -> list[list[float]] | ArrayTensor:
    """Pad/dilate bounding boxes by a percentage of their width and height."""
    to_list = False
    if isinstance(bboxes, (list, tuple)):
        bboxes = np.asarray(bboxes)
        image_sizes = np.asarray(image_sizes) if image_sizes is not None else None
        to_list = True

    module = module_from_obj(bboxes)
    if module.__name__ not in ["torch", "numpy"]:
        msg = "bboxes must be a list, tuple, numpy array or torch tensor"
        raise TypeError(msg)

    is_int = (module.__name__ == "torch" and not module.is_floating_point(bboxes)) or bboxes.dtype.kind == "i"

    if image_sizes is None:
        image_sizes = module.ones_like(bboxes[..., :2])

    if bboxes.shape[-1] != 4:
        msg = "bboxes must be shape (..., 4)"
        raise ValueError(msg)
    if image_sizes.shape[-1] != 2:
        msg = "image_sizes must be shape (..., 2)"
        raise ValueError(msg)
    if bboxes.shape[:-1] != image_sizes.shape[:-1]:
        msg = "bboxes and image_sizes must have the same number of dims and same size in each (apart from the last)"
        raise ValueError(msg)

    bbox_widths = bboxes[..., 2] - bboxes[..., 0]
    bbox_heights = bboxes[..., 3] - bboxes[..., 1]
    dx = bbox_widths * float(padding_perc)
    dy = bbox_heights * float(padding_perc)

    xmin = module.maximum(module.minimum(bboxes[..., 0], 0), module.floor(bboxes[..., 0] - dx))
    ymin = module.maximum(module.minimum(bboxes[..., 1], 0), module.floor(bboxes[..., 1] - dy))
    xmax = module.minimum(module.maximum(bboxes[..., 2], image_sizes[..., 0]), module.ceil(bboxes[..., 2] + dx))
    ymax = module.minimum(module.maximum(bboxes[..., 3], image_sizes[..., 1]), module.ceil(bboxes[..., 3] + dy))
    padded_bboxes = stack_along([xmin, ymin, xmax, ymax], dim=-1)

    if is_int:
        padded_bboxes = padded_bboxes.int() if module.__name__ == "torch" else padded_bboxes.astype(int)
    if to_list:
        return padded_bboxes.tolist()
    return padded_bboxes


def sanitize_bboxes(
    bboxes_xyxy: list | ArrayTensor, image_sizes_hw: list | ArrayTensor
) -> list[list[int]] | ArrayTensor:
    """Clip bbox coordinates to valid crop rectangles.

    Inputs use bbox format `(x1, y1, x2, y2)` and image size format `(H, W)`.
    `image_sizes_hw` can be shared across all bboxes (shape `(2,)`) or provided
    per bbox (shape `(..., 2)` matching `bboxes_xyxy[..., :2]`).
    """
    to_list = False
    if isinstance(bboxes_xyxy, (list, tuple)):
        bboxes_xyxy = np.asarray(bboxes_xyxy)
        image_sizes_hw = np.asarray(image_sizes_hw)
        to_list = True

    module = module_from_obj(bboxes_xyxy)
    if module.__name__ not in ["torch", "numpy"]:
        msg = "bboxes_xyxy must be a list, tuple, numpy array or torch tensor"
        raise TypeError(msg)

    if module_from_obj(image_sizes_hw).__name__ != module.__name__:
        if module.__name__ == "numpy":
            image_sizes_hw = np.asarray(image_sizes_hw)
        else:
            image_sizes_hw = module.as_tensor(image_sizes_hw, device=bboxes_xyxy.device)

    if bboxes_xyxy.shape[-1] != 4:
        msg = "bboxes_xyxy must be shape (..., 4)"
        raise ValueError(msg)
    if image_sizes_hw.shape[-1] != 2:
        msg = "image_sizes_hw must be shape (2,) or (..., 2)"
        raise ValueError(msg)

    if image_sizes_hw.shape[:-1] == ():
        image_sizes_hw = module.broadcast_to(image_sizes_hw, bboxes_xyxy[..., :2].shape)
    elif image_sizes_hw.shape[:-1] != bboxes_xyxy.shape[:-1]:
        msg = "image_sizes_hw must be shape (2,) or (..., 2) matching bboxes_xyxy leading dims"
        raise ValueError(msg)

    if module.any(image_sizes_hw <= 0):
        msg = "image_sizes_hw values must be strictly positive"
        raise ValueError(msg)

    image_heights = image_sizes_hw[..., 0]
    image_widths = image_sizes_hw[..., 1]

    x1 = module.clip(module.floor(bboxes_xyxy[..., 0]), 0, image_widths - 1)
    y1 = module.clip(module.floor(bboxes_xyxy[..., 1]), 0, image_heights - 1)
    x2 = module.maximum(module.clip(module.ceil(bboxes_xyxy[..., 2]), 0, image_widths), x1 + 1)
    y2 = module.maximum(module.clip(module.ceil(bboxes_xyxy[..., 3]), 0, image_heights), y1 + 1)

    sanitized_bboxes = cast_dtype(stack_along([x1, y1, x2, y2], dim=-1), "int64")
    if to_list:
        return sanitized_bboxes.tolist()
    return sanitized_bboxes


def _as_mask_array(masks: ArrayTensor | Image.Image | list[Any]) -> ArrayTensor:
    """Convert PIL inputs to arrays while preserving tensor inputs."""
    if isinstance(masks, Image.Image):
        array = np.asarray(masks)
        if array.ndim == 3 and array.shape[-1] != 1:
            msg = "PIL mask images must be single-channel (mode '1' or 'L')"
            raise ValueError(msg)
        return array

    if isinstance(masks, (list, tuple)):
        if not masks:
            msg = "masks list cannot be empty"
            raise ValueError(msg)

        arrays = [_as_mask_array(mask) for mask in masks]
        module = module_from_obj(arrays[0])
        if module.__name__ not in ["torch", "numpy"]:
            msg = "masks list items must resolve to numpy arrays or torch tensors"
            raise TypeError(msg)
        if any(module_from_obj(array).__name__ != module.__name__ for array in arrays):
            msg = "all masks in a list must use the same backend (numpy or torch)"
            raise TypeError(msg)

        return stack_along(arrays, 0)

    return masks


def masks_to_bboxes(masks: ArrayTensor | Image.Image | list[Image.Image]) -> ArrayTensor:
    """Compute tight bboxes (x_min, y_min, x_max, y_max) from binary masks.

    Accepted mask shapes are `[..., H, W]` and `[..., H, W, 1]`.
    Empty masks return `[0, 0, 0, 0]` for that entry.
    """
    masks = _as_mask_array(masks)

    module = module_from_obj(masks)
    if module.__name__ not in ["torch", "numpy"]:
        msg = "masks must be a numpy array, torch tensor, PIL image, or list of PIL images"
        raise TypeError(msg)

    if masks.ndim < 2:
        msg = "masks must have at least 2 dims and shape (..., H, W) or (..., H, W, 1)"
        raise ValueError(msg)

    if masks.shape[-1] == 1 and masks.ndim >= 3:
        masks = masks[..., 0]

    # Flatten leading batch dims so bbox extraction runs on a simple [N, H, W] view.
    leading_shape = tuple(masks.shape[:-2])
    height, width = masks.shape[-2:]
    flat_masks = (masks != 0).reshape(-1, height, width)

    # Reduce each mask to "which rows/cols contain foreground?"
    y_any = any_along(flat_masks, 2)
    x_any = any_along(flat_masks, 1)
    has_fg = any_along(y_any, 1)

    # Build row/column index grids for broadcasting with y_any/x_any.
    y_idx = module.arange(height).reshape(1, height)
    x_idx = module.arange(width).reshape(1, width)

    # Sentinel values make min/max robust:
    # - min candidates use (height/width) where no foreground exists
    # - max candidates use -1 where no foreground exists
    # Then we clamp empty masks back to zero below.
    y_min_candidates = module.where(y_any, y_idx, height)
    y_max_candidates = module.where(y_any, y_idx, -1)
    x_min_candidates = module.where(x_any, x_idx, width)
    x_max_candidates = module.where(x_any, x_idx, -1)

    y_min = min_along(y_min_candidates, dim=1)
    y_max = max_along(y_max_candidates, dim=1)
    x_min = min_along(x_min_candidates, dim=1)
    x_max = max_along(x_max_candidates, dim=1)

    # Empty mask -> [0, 0, 0, 0].
    zeros = module.zeros_like(x_min)
    x_min = module.where(has_fg, x_min, zeros)
    y_min = module.where(has_fg, y_min, zeros)
    x_max = module.where(has_fg, x_max, zeros)
    y_max = module.where(has_fg, y_max, zeros)

    bboxes = stack_along([x_min, y_min, x_max, y_max], -1)
    return cast_dtype(bboxes.reshape(*leading_shape, 4), "int64")
