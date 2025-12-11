"""Utilities for importing modules."""

from __future__ import annotations

import importlib
import sys
from functools import cache
from typing import TYPE_CHECKING, Any

from .logger import get_logger

if TYPE_CHECKING:
    import types

logger = get_logger(__name__)


def module_from_obj(x: Any) -> types.ModuleType:
    """Get the module for a given object."""
    top = type(x).__module__
    try:
        return importlib.import_module(top)
    except ImportError:
        # fallback: search loaded modules for the prefix
        for name, mod in sys.modules.items():
            if name == top or name.startswith(top + "."):
                return mod
    msg = f"couldn't load module for {type(x)}"
    raise ImportError(msg)


@cache
def _module_available(name: str) -> tuple[bool, ImportError | None]:
    """Check if a module is available."""
    try:
        importlib.import_module(name)
    except ImportError as e:
        return False, e
    else:
        return True, None


def module_available(
    module: str, log_warning: str | None = None, return_error: bool = False
) -> bool | tuple[bool, ImportError | None]:
    """Check if a module is available.

    Args:
        module (str): The name of the module to check.
        log_warning (str | None): Optional warning message to log if the module is not available.
        return_error (bool): Whether to return the ImportError if raised.

    Returns:
        bool | tuple[bool, ImportError | None]: True if the module is installed, False otherwise. If return_error is
            True, also returns the ImportError if raised or None.

    """
    available, error = _module_available(module)
    if error is not None and log_warning is not None:
        logger.warning(log_warning)

    if return_error:
        return available, error
    return available


def requires(package: str | list[str], msg: str | None = None) -> None:
    """Check if a package or list of packages is available and raise ImportError if not.

    Args:
        package (str | list[str]): The name of the package to check.
        msg (str | None): Optional custom error message to raise if the package is not available.

    Raises:
        ImportError: If the package is not available.

    """
    package = [package] if isinstance(package, str) else package
    available = [module_available(p) for p in package]
    if not all(available):
        missing = [p for p, inst in zip(package, available, strict=True) if not inst]
        pre_msg = f"Required package(s) {', '.join(missing)} not available."
        raise ImportError(pre_msg if msg is None or msg.strip() == "" else pre_msg + msg)
