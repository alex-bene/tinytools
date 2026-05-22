"""Backend-agnostic array/tensor operations for numpy and torch."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

from .imports import module_from_obj

if TYPE_CHECKING:
    from types import ModuleType

    import numpy as np
    import torch  # pyright: ignore[reportMissingImports]

    ArrayTensor: TypeAlias = np.ndarray | torch.Tensor


def all_along(x: ArrayTensor, dim: int) -> ArrayTensor:
    """Apply `all` along a dimension for numpy arrays or torch tensors."""
    is_numpy = module_from_obj(x).__name__ == "numpy"
    return x.all(**{("axis" if is_numpy else "dim"): dim})


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


def numel(x: ArrayTensor) -> ArrayTensor:
    """Get the number of elements for numpy arrays or torch tensors."""
    module = module_from_obj(x)
    if module.__name__ == "numpy":
        return x.size
    return x.numel()


def atan(x: ArrayTensor) -> ArrayTensor:
    """Get the number of elements for numpy arrays or torch tensors."""
    module = module_from_obj(x)
    if module.__name__ == "numpy":
        return module.arctan(x)
    return module.atan(x)


def get_device(x: ArrayTensor) -> torch.device | None:
    """Get the number of elements for numpy arrays or torch tensors."""
    module = module_from_obj(x)
    if module.__name__ == "numpy":
        return None
    return x.device


def arraytensor(x: Any, dtype: None = None, *, module: ModuleType, **kwargs) -> ArrayTensor:
    """Create a numpy array or torch tensor."""
    tensor_only_kwargs = ("device", "requires_grad", "pin_memory")
    if module.__name__ == "numpy":
        kwargs = {k: v for k, v in kwargs.items() if k not in tensor_only_kwargs}
        return module.array(x, dtype=dtype, **kwargs)

    array_only_kwargs = ("copy", "order", "subok", "ndmin", "like")
    kwargs = {k: v for k, v in kwargs.items() if k not in array_only_kwargs}
    return module.tensor(x, dtype=dtype, **kwargs)


def cast_dtype(x: ArrayTensor, dtype: str | torch.dtype | np.dtype, copy: bool | None = None) -> ArrayTensor:
    """Cast arrays/tensors to a module dtype by name (e.g. 'int64', 'float32'), torch dtype or numpy dtype."""
    module = module_from_obj(x)
    if isinstance(dtype, str):
        dtype = getattr(module, dtype)
    if module.__name__ == "numpy":
        return x.astype(dtype, copy=True if copy is None else copy)
    return x.to(dtype=dtype, copy=False if copy is None else copy)


def move_device(x: ArrayTensor, device: str) -> ArrayTensor:
    """Move arrays/tensors to a device by name (e.g. 'cpu', 'cuda')."""
    module = module_from_obj(x)
    if module.__name__ == "numpy":
        return x
    return x.to(device=module.device(device))


def deg2rad(x: ArrayTensor) -> ArrayTensor:
    """Convert degrees to radians for numpy arrays or torch tensors."""
    module = module_from_obj(x)
    return module.deg2rad(x)


def rad2deg(x: ArrayTensor) -> ArrayTensor:
    """Convert radians to degrees for numpy arrays or torch tensors."""
    module = module_from_obj(x)
    return module.rad2deg(x)


def squeeze_along(x: ArrayTensor, dim: int) -> ArrayTensor:
    """Squeeze along a dimension for numpy arrays or torch tensors."""
    module = module_from_obj(x)
    return module.squeeze(x, **{("axis" if module.__name__ == "numpy" else "dim"): dim})


def unsqueeze_along(x: ArrayTensor, dim: int) -> ArrayTensor:
    """Unsqueeze along a dimension for numpy arrays or torch tensors."""
    module = module_from_obj(x)
    if module.__name__ == "numpy":
        return module.expand_dims(x, axis=dim)
    return module.unsqueeze(x, dim=dim)


def squeeze_to(x: ArrayTensor, ndim: int, dim: int = 0) -> ArrayTensor:
    """Squeeze dimension `dim` until the ArrayTensor reaches the target rank."""
    target_ndim = int(ndim)
    if x.ndim < target_ndim:
        msg = f"Cannot squeeze from rank {x.ndim} to larger rank {target_ndim}."
        raise ValueError(msg)
    dim = int(dim)
    if dim < 0:
        dim += x.ndim
    if dim >= x.ndim:
        msg = f"Cannot squeeze dimension {dim} from rank {x.ndim}."
        raise ValueError(msg)

    output = x
    while output.ndim > target_ndim:
        if output.shape[dim] != 1:
            msg = (
                f"Cannot squeeze dimension {dim} with size {output.shape[dim]} "
                f"to reach rank {target_ndim} from rank {output.ndim}."
            )
            raise ValueError(msg)
        output = squeeze_along(output, dim)
    return output


def unsqeeze_to(x: ArrayTensor, ndim: int, dim: int = 0) -> ArrayTensor:
    """Unsqueeze dimension `dim` until the ArrayTensor reaches the target rank."""
    target_ndim = int(ndim)
    if x.ndim < target_ndim:
        msg = f"Cannot unsqueeze from rank {x.ndim} to larger rank {target_ndim}."
        raise ValueError(msg)
    dim = int(dim)
    if dim < 0:
        dim += x.ndim
    if dim >= x.ndim:
        msg = f"Cannot unsqueeze dimension {dim} from rank {x.ndim}."
        raise ValueError(msg)

    output = x
    while output.ndim < target_ndim:
        output = unsqueeze_along(output, dim)
    return output


def is_integer(x: ArrayTensor) -> ArrayTensor:
    """Check if an array/tensor is integer type."""
    module = module_from_obj(x)
    if module.__name__ == "numpy":
        return module.issubdtype(x.dtype, module.integer)
    return not (x.is_floating_point() or x.is_complex() or x.dtype == module.bool)
