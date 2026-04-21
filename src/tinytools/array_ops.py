"""Backend-agnostic array/tensor operations for numpy and torch."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

from .imports import module_from_obj

if TYPE_CHECKING:
    import numpy as np
    from torch import Tensor  # pyright: ignore[reportMissingImports]

    ArrayTensor: TypeAlias = np.ndarray | Tensor


def any_along(x: ArrayTensor, dim: int) -> ArrayTensor:
    """Apply `any` along a dimension for numpy arrays or torch tensors."""
    is_numpy = module_from_obj(x).__name__ == "numpy"
    return x.any(**{("axis" if is_numpy else "dim"): dim})


def argmax_along(x: ArrayTensor, dim: int) -> ArrayTensor:
    """Apply `argmax` along a dimension for numpy arrays or torch tensors."""
    is_numpy = module_from_obj(x).__name__ == "numpy"
    return x.argmax(**{("axis" if is_numpy else "dim"): dim})


def max_along(x: ArrayTensor, dim: int) -> ArrayTensor:
    """Apply `max` along a dimension for numpy arrays or torch tensors."""
    is_numpy = module_from_obj(x).__name__ == "numpy"
    output = x.max(**{("axis" if is_numpy else "dim"): dim})
    return output.values if hasattr(output, "values") else output


def min_along(x: ArrayTensor, dim: int) -> ArrayTensor:
    """Apply `min` along a dimension for numpy arrays or torch tensors."""
    is_numpy = module_from_obj(x).__name__ == "numpy"
    output = x.min(**{("axis" if is_numpy else "dim"): dim})
    return output.values if hasattr(output, "values") else output


def flip_along(x: ArrayTensor, dim: int) -> ArrayTensor:
    """Flip along a dimension for numpy arrays or torch tensors."""
    module = module_from_obj(x)
    flip_dim = dim if module.__name__ == "numpy" else [dim]
    return module.flip(x, **{("axis" if module.__name__ == "numpy" else "dims"): flip_dim})


def stack_along(xs: list[ArrayTensor], dim: int) -> ArrayTensor:
    """Stack arrays/tensors along a dimension for numpy arrays or torch tensors."""
    if not xs:
        msg = "xs cannot be empty"
        raise ValueError(msg)
    module = module_from_obj(xs[0])
    return module.stack(xs, **{("axis" if module.__name__ == "numpy" else "dim"): dim})


def cast_dtype(x: ArrayTensor, dtype_name: str) -> ArrayTensor:
    """Cast arrays/tensors to a module dtype by name (e.g. 'int64', 'float32')."""
    module = module_from_obj(x)
    dtype = getattr(module, dtype_name)
    if module.__name__ == "numpy":
        return x.astype(dtype)
    return x.to(dtype=dtype)
