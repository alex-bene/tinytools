"""Bounding box tensor conversion helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tinytools.imports import optional_module
from tinytools.torch.utils import as_float_tensor
from tinytools.validate import validate_ndim, validate_shape

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch  # pyright: ignore[reportMissingImports]

    from tinytools.array_ops import ArrayTensor

    BBoxInput = Sequence[Sequence[tuple[float, float, float, float]] | ArrayTensor | None] | ArrayTensor
else:
    torch = optional_module("torch", extra="torch")


def prepare_bboxes(
    bboxes: BBoxInput | None, *, device: torch.device | None = None, allow_none: bool = False
) -> list[torch.Tensor | None] | None:
    """Validate bounding-box inputs and convert them to tensors.

    Args:
        bboxes (BBoxInput | None): Input bounding boxes as `(x0, y0, x1, y1)` vectors.
            Shape: [(N, 4), ...]
        device (torch.device | None, optional): Target device for output tensors.
            If `None`, tensors remain on their current/default device. Default: None.
        allow_none (bool, optional): Whether `None` is accepted for the top-level input
            and preserved inside batch sequences. Default: False.

    Returns:
        list[torch.Tensor | None] | None: Bounding boxes as float tensors on `device`.
            Shape: [(N, 4) | None, ...]

    """
    if bboxes is None:
        if allow_none:
            return None
        msg = "`bboxes` cannot be None unless `allow_none=True`."
        raise ValueError(msg)

    bboxes = list(bboxes)

    result: list[torch.Tensor] = []
    for bboxes_i in bboxes:
        if bboxes_i is None and allow_none:
            result.append(None)
            continue
        if bboxes_i is None:
            msg = "`bboxes` cannot be None unless `allow_none=True`."
            raise ValueError(msg)
        new_bboxes_i = bboxes_i
        if not isinstance(bboxes_i, (np.ndarray, torch.Tensor)):
            new_bboxes_i = np.asarray(bboxes_i, dtype=np.float32)
        new_bboxes_i = as_float_tensor(new_bboxes_i, device=device)
        validate_ndim(new_bboxes_i, ndim=2, arg_name="nested bboxes")
        validate_shape(new_bboxes_i, shape=[-1, 4], arg_name="nested bboxes")
        result.append(new_bboxes_i)

    return result
