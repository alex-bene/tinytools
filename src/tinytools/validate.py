"""Array validation helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import EllipsisType

    from .array_ops import ArrayTensor


T = TypeVar("T")


def validate_shape(array: ArrayTensor, shape: tuple[int | None | EllipsisType, ...], arg_name: str = "array") -> None:
    """Validate that an array shape matches a shape specification.

    Args:
        array (torch.Tensor | np.ndarray): Tensor/array to validate.
        shape (tuple[int | None | EllipsisType, ...]): Expected shape spec.
            Use an `int` for an exact size, `None` or `< 0` for any size at that
            axis, and `...` for any number of middle axes.
            Shape: (...,)
        arg_name (str, optional): Name used in error messages. Default: "array".

    Returns:
        None: This function returns nothing.

    """
    if shape.count(...) > 1:
        msg = f"`shape` may include at most one ellipsis, got {shape}."
        raise ValueError(msg)

    if ... not in shape:
        _validate_shape_parts(
            actual_shape=tuple(array.shape), expected_parts=shape, offset=0, arg_name=arg_name, full_expected=shape
        )
        return

    ellipsis_index = shape.index(...)
    prefix = shape[:ellipsis_index]
    suffix = shape[ellipsis_index + 1 :]
    required_rank = len(prefix) + len(suffix)
    if array.ndim < required_rank:
        msg = (
            f"Expected `{arg_name}` to have rank >= {required_rank} for shape spec {shape}, "
            f"got shape {tuple(array.shape)}."
        )
        raise ValueError(msg)

    _validate_shape_parts(
        actual_shape=tuple(array.shape[: len(prefix)]),
        expected_parts=prefix,
        offset=0,
        arg_name=arg_name,
        full_expected=shape,
    )
    _validate_shape_parts(
        actual_shape=tuple(array.shape[array.ndim - len(suffix) :]),
        expected_parts=suffix,
        offset=array.ndim - len(suffix),
        arg_name=arg_name,
        full_expected=shape,
    )


def validate_ndim(array: ArrayTensor, ndim: int | tuple[int, ...], arg_name: str = "array") -> None:
    """Validate that an array rank matches the expected rank.

    Args:
        array (torch.Tensor | np.ndarray): Tensor/array to validate.
        ndim (int | tuple[int, ...]): Allowed rank(s).
        arg_name (str, optional): Name used in error messages. Default: "array".

    Returns:
        None: This function returns nothing.

    """
    allowed_ranks = (ndim,) if isinstance(ndim, int) else ndim
    if array.ndim not in allowed_ranks:
        msg = (
            f"Expected `{arg_name}` rank to be one of {allowed_ranks}, "
            f"got rank {array.ndim} with shape {tuple(array.shape)}."
        )
        raise ValueError(msg)


def validate_same_shape(
    array_a: ArrayTensor, array_b: ArrayTensor, arg_name_a: str = "array_a", arg_name_b: str = "array_b"
) -> None:
    """Validate that two arrays have identical shapes.

    Args:
        array_a (torch.Tensor | np.ndarray): First tensor/array.
        array_b (torch.Tensor | np.ndarray): Second tensor/array.
        arg_name_a (str, optional): First array name for error messages. Default: "array_a".
        arg_name_b (str, optional): Second array name for error messages. Default: "array_b".

    Returns:
        None: This function returns nothing.

    """
    if tuple(array_a.shape) != tuple(array_b.shape):
        msg = (
            f"Expected `{arg_name_a}` and `{arg_name_b}` to have the same shape, "
            f"got {tuple(array_a.shape)} and {tuple(array_b.shape)}."
        )
        raise ValueError(msg)


def validate_same_size(
    array_a: ArrayTensor,
    array_b: ArrayTensor,
    *,
    dim_a: int,
    dim_b: int | None = None,
    arg_name_a: str = "array_a",
    arg_name_b: str = "array_b",
) -> None:
    """Validate that two array dimensions have the same size.

    Args:
        array_a (torch.Tensor | np.ndarray): First tensor/array.
        array_b (torch.Tensor | np.ndarray): Second tensor/array.
        dim_a (int): Dimension index in `array_a`.
        dim_b (int | None, optional): Dimension index in `array_b`. If None, uses `dim_a`.
            Default: None.
        arg_name_a (str, optional): First array name for error messages. Default: "array_a".
        arg_name_b (str, optional): Second array name for error messages. Default: "array_b".

    Returns:
        None: This function returns nothing.

    """
    if dim_b is None:
        dim_b = dim_a
    if array_a.shape[dim_a] != array_b.shape[dim_b]:
        msg = (
            f"Expected `{arg_name_a}.shape[{dim_a}]` and `{arg_name_b}.shape[{dim_b}]` to match, "
            f"got {array_a.shape[dim_a]} and {array_b.shape[dim_b]}."
        )
        raise ValueError(msg)


def validate_equal_length(
    sequence_a: Sequence[object],
    sequence_b: Sequence[object],
    arg_name_a: str = "sequence_a",
    arg_name_b: str = "sequence_b",
) -> None:
    """Validate that two sequences have equal length.

    Args:
        sequence_a (Sequence[object]): First sequence to validate.
        sequence_b (Sequence[object]): Second sequence to validate.
        arg_name_a (str, optional): First sequence name for error messages.
            Default: "sequence_a".
        arg_name_b (str, optional): Second sequence name for error messages.
            Default: "sequence_b".

    Returns:
        None: This function returns nothing.

    """
    if len(sequence_a) != len(sequence_b):
        msg = (
            f"Expected `{arg_name_a}` and `{arg_name_b}` to have the same length, "
            f"got {len(sequence_a)} and {len(sequence_b)}."
        )
        raise ValueError(msg)


def validate_in_range(
    value: T,
    *,
    min_value: T | None = None,
    max_value: T | None = None,
    include_min: bool = True,
    include_max: bool = True,
    arg_name: str = "value",
) -> None:
    """Validate that a comparable value is within a range.

    Args:
        value (T): Value to validate.
        min_value (T | None, optional): Lower bound. If None, no lower bound is
            applied. Default: None.
        max_value (T | None, optional): Upper bound. If None, no upper bound is
            applied. Default: None.
        include_min (bool, optional): If True, enforce `value >= min_value`. If
            False, enforce `value > min_value`. Default: True.
        include_max (bool, optional): If True, enforce `value <= max_value`. If
            False, enforce `value < max_value`. Default: True.
        arg_name (str, optional): Value name for error messages. Default: "value".

    Returns:
        None: This function returns nothing.

    """
    if min_value is None and max_value is None:
        msg = "At least one of `min_value` or `max_value` must be provided."
        raise ValueError(msg)
    if min_value is not None and max_value is not None and min_value > max_value:
        msg = f"Invalid range for `{arg_name}`: min_value ({min_value}) cannot be greater than max_value ({max_value})."
        raise ValueError(msg)
    if min_value is not None:
        is_below_min = value < min_value if include_min else value <= min_value
        if is_below_min:
            relation = ">=" if include_min else ">"
            msg = f"`{arg_name}` must be {relation} {min_value}, got {value}."
            raise ValueError(msg)
    if max_value is not None:
        is_above_max = value > max_value if include_max else value >= max_value
        if is_above_max:
            relation = "<=" if include_max else "<"
            msg = f"`{arg_name}` must be {relation} {max_value}, got {value}."
            raise ValueError(msg)


def validate_all_instances(
    values: Sequence[object] | None,
    expected_type: type[object] | tuple[type[object], ...],
    arg_name: str = "values",
    *,
    allow_none: bool = False,
) -> None:
    """Validate that every sequence item is an instance of the expected type.

    Args:
        values (Sequence[object] | None): Sequence of values to validate. If None,
            validation passes only when `allow_none` is True.
        expected_type (type[object] | tuple[type[object], ...]): Allowed type(s)
            for non-None entries.
        arg_name (str, optional): Sequence name for error messages. Default: "values".
        allow_none (bool, optional): Whether `None` entries are permitted. Default: False.

    Returns:
        None: This function returns nothing.

    """
    if values is None:
        if allow_none:
            return
        msg = f"`{arg_name}` must be a sequence, got None."
        raise TypeError(msg)

    is_valid = all((value is None and allow_none) or isinstance(value, expected_type) for value in values)
    if is_valid:
        return
    if allow_none:
        msg = f"`{arg_name}` must contain only instances of {expected_type} or None."
        raise TypeError(msg)
    msg = f"`{arg_name}` must contain only instances of {expected_type}."
    raise TypeError(msg)


def validate_and_fill_optional_list(
    values: list[T] | None, expected_length: int, arg_name: str = "values", fill_value: T | None = None
) -> list[T | None]:
    """Normalize an optional list to a fixed length.

    Args:
        values (list[T] | None): List to normalize. If None, a new list filled with
            `fill_value` is returned.
        expected_length (int): Required output length.
        arg_name (str, optional): Argument name used in error messages. Default: "values".
        fill_value (T | None, optional): Fill value used when `values` is None.
            Default: None.

    Returns:
        list[T | None]: Normalized list. Shape: (expected_length,)

    """
    if values is None:
        return [fill_value] * expected_length
    if len(values) != expected_length:
        msg = f"Expected {arg_name} to have length {expected_length}, got {len(values)}."
        raise ValueError(msg)
    return values


def _validate_shape_parts(
    *,
    actual_shape: tuple[int, ...],
    expected_parts: tuple[int | None | EllipsisType, ...],
    offset: int,
    arg_name: str,
    full_expected: tuple[int | None | EllipsisType, ...],
) -> None:
    if len(actual_shape) != len(expected_parts):
        msg = (
            f"Internal shape validation error for `{arg_name}` with expected spec {full_expected}: "
            f"got rank slice {actual_shape} against expected part {expected_parts}."
        )
        raise ValueError(msg)
    for local_dim, expected_dim in enumerate(expected_parts):
        if expected_dim is None or expected_dim < 0:
            continue
        if expected_dim is ...:
            msg = f"Unexpected ellipsis in shape part for `{arg_name}`."
            raise ValueError(msg)
        actual_dim = actual_shape[local_dim]
        if actual_dim != expected_dim:
            global_dim = offset + local_dim
            msg = (
                f"Expected `{arg_name}` to match shape spec {full_expected}, got shape with "
                f"mismatch at dim {global_dim}: expected {expected_dim}, got {actual_dim}."
            )
            raise ValueError(msg)


def validate_lengths(
    *, expected_length: int, allow_none: bool | Sequence[str] = False, **kwargs: Sequence[Any]
) -> None:
    """Validate top-level lengths for flattened input preparation."""
    for arg_name, value in kwargs.items():
        allow_none_c = allow_none if isinstance(allow_none, bool) else (arg_name in allow_none)
        if allow_none_c and (value is None):
            return
        if value is None:
            msg = f"{arg_name} must be provided and not None."
            raise ValueError(msg)
        if len(value) != expected_length:
            msg = f"Expected len({arg_name}) == len(images), got {len(value)} and {expected_length}."
            raise ValueError(msg)
