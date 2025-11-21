"""Metaclasses for tinytools."""

from typing import Any


class FrozenNamespaceMeta(type):
    """Metaclass to support dict-style access and immutability."""

    def __getitem__(cls, key: str) -> Any:
        """Get attribute by name."""
        return getattr(cls, key)

    def __setattr__(cls, key: str, value: Any) -> None:
        """Prevent modification of attributes."""
        msg = f"{cls.__name__} is immutable."
        raise TypeError(msg)

    def __delattr__(cls, _: str) -> None:
        """Prevent deletion of attributes."""
        msg = f"{cls.__name__} is immutable."
        raise TypeError(msg)

    def __call__(cls, *_: Any, **__: Any) -> Any:
        """Prevent instantiation."""
        msg = f"{cls.__name__} cannot be instantiated."
        raise TypeError(msg)
