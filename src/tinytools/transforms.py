"""Data transformation tools."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from .imports import module_available

with contextlib.suppress(ImportError):
    import torch

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any


def resize(  # noqa: PLR0912
    image: Image.Image,
    size: tuple[int, int],
    depth: np.ndarray | Tensor | None = None,
    bounding_boxes: np.ndarray | Tensor | None = None,
    points: np.ndarray | Tensor | None = None,
    resample: int | None = None,
) -> dict[str, tuple[int, int, int, int] | tuple[int, int]]:
    """Resize annotation based on old and new image sizes.

    Args:
        image (PIL.Image.Image): Image to resize.
        size (tuple[int, int]): New image size (width, height).
        depth (np.ndarray | torch.Tensor | None, optional): Depth map of shape (H, W). Defaults to None.
        bounding_boxes (np.ndarray | torch.Tensor | None, optional): Bounding boxes coordinates
            (x_min, y_min, x_max, y_max). Defaults to None.
        points (np.ndarray | torch.Tensor | None, optional): Points coordinates. Defaults to None.
        resample (int | None, optional): Resampling method. Defaults to None.

    Returns:
        dict[str, Image.Image | np.ndarray | torch.Tensor]: Resized image and annotations.

    """
    # Get original dimensions
    if isinstance(image, Image.Image):
        old_width, old_height = image.size
    else:
        msg = "image must be PIL.Image.Image"
        raise TypeError(msg)
    new_width, new_height = size

    # Calculate resize ratios
    width_ratio = new_width / old_width
    height_ratio = new_height / old_height

    resized_annotations = {}
    # Scale bounding box coordinates
    if (
        not module_available("torch", log_warning="`torch` is required for depth interpolation. Return depth as None.")
        and depth is not None
    ):
        resized_annotations["depth"] = None
    elif depth is not None:
        # interpolate
        if isinstance(depth, np.ndarray):
            torch_depth = torch.from_numpy(depth)
        elif torch.is_tensor(depth):
            torch_depth = depth
        else:
            msg = "depth must be either np.ndarray or torch.Tensor"
            raise TypeError(msg)
        resized_torch_depth = torch.nn.functional.interpolate(
            torch_depth.unsqueeze(0).unsqueeze(0),
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
            antialias=False,
        )[0][0]
        if torch.is_tensor(depth):
            resized_annotations["depth"] = resized_torch_depth
        else:
            resized_annotations["depth"] = resized_torch_depth.cpu().numpy()

    # Scale bounding box coordinates
    if bounding_boxes is not None:
        if bounding_boxes.ndim != 2 or bounding_boxes.shape[1] != 4:
            msg = "bounding_boxes must be a 2D array with shape (N, 4)"
            raise ValueError(msg)
        if isinstance(bounding_boxes, np.ndarray):
            scale = np.array([width_ratio, height_ratio, width_ratio, height_ratio])[None, :]
            resized_annotations["bounding_boxes"] = (bounding_boxes * scale).astype(int)
        elif torch.is_tensor(bounding_boxes):
            scale = torch.tensor([width_ratio, height_ratio, width_ratio, height_ratio], device=bounding_boxes.device)
            resized_annotations["bounding_boxes"] = (bounding_boxes * scale).int()
        else:
            msg = "bounding_boxes must be either np.ndarray or torch.Tensor"
            raise TypeError(msg)

    # Scale point coordinates
    if points is not None:
        if points.ndim != 2 or points.shape[1] != 2:
            msg = "points must be a 2D array with shape (N, 2)"
            raise ValueError(msg)
        if isinstance(points, np.ndarray):
            resized_annotations["point"] = (points * np.array([width_ratio, height_ratio])[None, :]).astype(int)
        elif torch.is_tensor(points):
            resized_annotations["point"] = (
                points * torch.tensor([width_ratio, height_ratio], device=points.device)
            ).int()
        else:
            msg = "points must be either np.ndarray or torch.Tensor"
            raise TypeError(msg)

    # Resize image
    resized_annotations["image"] = image.resize(size, resample=resample)

    return resized_annotations
