"""Input preparation utility functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import numpy as np

from tinytools.imports import optional_module

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch  # pyright: ignore[reportMissingImports]

else:
    torch = optional_module("torch", extra="torch")

T = TypeVar("T")


def is_batch_container(value: object, single_ndim: int) -> bool:
    """Return whether value is a top-level container of per-sample inputs."""
    if isinstance(value, (np.ndarray, torch.Tensor)) and value.ndim == (single_ndim + 1):
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


def expand_to_batch(value: Sequence[T | None] | T | None, *, batch_size: int, single_ndim: int) -> list[T | None]:
    """Expand scalar or None input to a batch-aligned list."""
    if is_batch_container(value, single_ndim=single_ndim):
        return list(value)
    return [value] * batch_size
