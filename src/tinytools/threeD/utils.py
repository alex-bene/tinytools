"""Utility functions for 3D operations."""

from __future__ import annotations

import torch


def get_scale_and_shift(
    pointmap: torch.Tensor, shift_only_z: bool = True, single_scale_per_pointmap: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute scale and shift for a pointmap.

    Args:
        pointmap: A tensor of shape (..., N, 3) representing the N points of a 3D pointmap.
        shift_only_z: If True, only compute shift along the Z-axis. Defaults to True.
        single_scale_per_pointmap: If True, compute a single scale value for the entire pointmap instead of one per
            axis. Defaults to True.

    Returns:
        A tuple containing:
            - scale: A tensor of shape (..., 3) representing the scale along each axis for each pointmap.
            - shift: A tensor of shape (..., 3) representing the shift along each axis for each pointmap.

    """
    if shift_only_z:
        shift_z = pointmap[..., -1].nanmedian(dim=-1, keepdim=True).values  # (..., 1)
        shift = torch.zeros_like(shift_z.expand(*shift_z.shape[:-1], 3))  # (..., 3)
        shift[..., -1] = shift_z
    else:
        shift = pointmap.nanmedian(dim=-2).values  # (..., 3)

    shifted_pointmap = pointmap - shift.unsqueeze(-2)  # (..., N, 3)
    scale = shifted_pointmap.abs().nanmean(dim=-2)  # (..., 3)
    if single_scale_per_pointmap:
        scale = scale.nanmean(dim=-1, keepdim=True).expand(*scale.shape)  # (..., 3)

    return scale, shift
