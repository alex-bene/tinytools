"""Pointmap tensor conversion helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from transformers.image_utils import ChannelDimension, infer_channel_dimension_format

from tinytools.imports import optional_module
from tinytools.threeD import CoordinateConversions
from tinytools.torch.utils import as_float_tensor
from tinytools.validate import validate_ndim, validate_shape

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch  # pyright: ignore[reportMissingImports]

    from tinytools.array_ops import ArrayTensor
else:
    torch = optional_module("torch", extra="torch")


def _get_coordinate_transform(
    *,
    input_convention: Literal["opencv", "opengl", "pytorch3d"],
    output_convention: Literal["opencv", "opengl", "pytorch3d"],
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build a row-vector transform matrix between two coordinate conventions.

    Args:
        input_convention (Literal["opencv", "opengl", "pytorch3d"]): Source coordinate convention.
        output_convention (Literal["opencv", "opengl", "pytorch3d"]): Target coordinate convention.
        dtype (torch.dtype): Data type for the returned transform.
        device (torch.device | None, optional): Device for the returned transform.

    Returns:
        Tensor: Transform matrix for row-vector multiplication. Shape: (3, 3)

    """
    if input_convention == output_convention:
        return torch.eye(3, device=device, dtype=dtype)

    input_name = {"opencv": "cv", "opengl": "opengl", "pytorch3d": "pt3d"}[input_convention]
    output_name = {"opencv": "cv", "opengl": "opengl", "pytorch3d": "pt3d"}[output_convention]
    transform = getattr(CoordinateConversions, f"{input_name}_to_{output_name}")
    return torch.tensor(transform, device=device, dtype=dtype)


def prepare_pointmaps(
    pointmaps: Sequence[ArrayTensor | None] | ArrayTensor | None,
    *,
    input_convention: Literal["opencv", "opengl", "pytorch3d"] = "opencv",
    output_convention: Literal["opencv", "opengl", "pytorch3d"] = "opencv",
    device: torch.device | None = None,
    allow_none: bool = False,
) -> list[torch.Tensor | None] | None:
    """Validate pointmap inputs and convert them to tensors.

    Args:
        pointmaps (Sequence[ndarray | Tensor | None] | ndarray | Tensor): Input pointmaps.
            Shape: [(H, W, 3) or (3, H, W), ...]
        input_convention (Literal["opencv", "opengl", "pytorch3d"], optional): Coordinate convention of
            input pointmaps. Default: "opencv".
        output_convention (Literal["opencv", "opengl", "pytorch3d"], optional): Coordinate convention of returned
            pointmaps. Default: "opencv".
        device (torch.device | None, optional): Target device for output tensors.
        allow_none (bool, optional): Allow None values in the input sequence. Default: False.

    Returns:
        list[Tensor | None] | None: Pointmaps as float32 tensors on `device`. Shape: [(H, W, 3) | None, ...]

    """
    if pointmaps is None:
        if allow_none:
            return None
        msg = "`pointmaps` cannot be None unless `allow_none=True`."
        raise ValueError(msg)

    if hasattr(pointmaps, "ndim") and pointmaps.ndim == 3:
        pointmaps = [pointmaps]
    pointmaps_list = list(pointmaps)
    transform = _get_coordinate_transform(
        input_convention=input_convention, output_convention=output_convention, device=device, dtype=torch.float32
    )

    result: list[torch.Tensor] = []
    for pointmap in pointmaps_list:
        if allow_none and pointmap is None:
            result.append(None)
            continue
        pm_tensor = as_float_tensor(pointmap, device=device)

        if infer_channel_dimension_format(pm_tensor) == ChannelDimension.FIRST:
            pm_tensor = pm_tensor.permute(1, 2, 0).contiguous()

        validate_ndim(pm_tensor, ndim=3, arg_name="pointmaps")
        validate_shape(pm_tensor, shape=[..., 3], arg_name="pointmaps")

        if input_convention != output_convention:
            transform = transform.to(pm_tensor)
            pm_tensor = pm_tensor @ transform
        result.append(pm_tensor)
    return result
