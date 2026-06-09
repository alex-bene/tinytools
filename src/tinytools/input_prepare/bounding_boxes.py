"""Bounding box tensor conversion helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tinytools.imports import optional_module
from tinytools.torch.utils import as_float_tensor
from tinytools.validate import validate_ndim, validate_shape

from .utils import expand_to_batch

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch  # pyright: ignore[reportMissingImports]

    from tinytools.array_ops import ArrayTensor

    BBox = tuple[int, int, int, int] | tuple[float, float, float, float] | ArrayTensor
    BBoxInput = Sequence[BBox | None] | BBox
else:
    torch = optional_module("torch", extra="torch")


def prepare_bboxes(
    bboxes: BBoxInput | None, *, device: torch.device | None = None, allow_none: bool = False
) -> list[torch.Tensor | None] | None:
    """Validate bounding-box inputs and convert them to tensors.

    Args:
        bboxes (BBoxInput | None): Input bounding boxes as `(x0, y0, x1, y1)` vectors.
            Shape: [(4,) | None, ...] or (4,)
        device (torch.device | None, optional): Target device for output tensors.
            If `None`, tensors remain on their current/default device. Default: None.
        allow_none (bool, optional): Whether `None` is accepted for the top-level input
            and preserved inside batch sequences. Default: False.

    Returns:
        list[torch.Tensor | None] | None: Bounding boxes as float tensors on `device`.
            Shape: [(4,) | None, ...]

    """
    if bboxes is None:
        if allow_none:
            return None
        msg = "`bboxes` cannot be None unless `allow_none=True`."
        raise ValueError(msg)

    bboxes_list = expand_to_batch(bboxes, single_ndim=1, batch_size=1)

    result: list[torch.Tensor] = []
    for bbox in bboxes_list:
        if allow_none and bbox is None:
            result.append(None)
            continue
        bbox_tensor = as_float_tensor(bbox, device=device)
        validate_ndim(bbox_tensor, ndim=1, arg_name="bboxes")
        validate_shape(bbox_tensor, shape=[4], arg_name="bboxes")
        if device is not None:
            bbox_tensor = bbox_tensor.to(device)
        result.append(bbox_tensor)
    return result
