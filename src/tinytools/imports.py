"""Utilities for importing modules."""

from __future__ import annotations

import importlib
import sys
from importlib import util as importlib_util
from typing import TYPE_CHECKING, Any

from .logger import get_logger

if TYPE_CHECKING:
    import types

logger = get_logger(__name__)

DEFAULT_OPTIONAL_DEPENDENCY_MESSAGE = (
    'Optional dependency "{package}" is not available. Install the required dependencies with: '
    'pip install "tinytools[{extra}]"'
)


class MissingOptionalDependency:
    """Proxy for an unavailable optional dependency.

    The proxy lets modules declare optional imports at module scope without assigning imported names
    to ``None``. Any actual use raises a clear ImportError.

    Args:
        package (str): Missing top-level package name.
        name (str | None, optional): Fully-qualified object name represented by this proxy.
            Default: None.
        extra (str, optional): Package extra to show in the installation hint. Default: "3d".
        message (str | None, optional): Custom error message. Default: None.

    """

    def __init__(self, package: str, name: str | None = None, *, extra: str = "3d", message: str | None = None) -> None:
        self.package = package
        self.name = name or package
        self.extra = extra
        self.message = message

    def _raise(self) -> None:
        """Raise a helpful ImportError for the missing dependency."""
        base = DEFAULT_OPTIONAL_DEPENDENCY_MESSAGE.format(package=self.package, extra=self.extra)
        suffix = f" Needed for: {self.name}." if self.name != self.package else ""
        raise ImportError(self.message or base + suffix)

    def __getattr__(self, _: str) -> Any:
        """Raise when an attribute on the missing dependency is accessed."""
        self._raise()

    def __call__(self, *_: Any, **__: Any) -> Any:
        """Raise when the missing dependency proxy is called."""
        self._raise()

    def __bool__(self) -> bool:
        """Return False for availability checks."""
        return False

    def __repr__(self) -> str:
        """Return a concise representation of the missing dependency."""
        return f"<MissingOptionalDependency {self.name!r}>"


def module_name_from_obj(x: Any) -> str:
    """Get the module name for a given object."""
    return type(x).__module__


def load_module(module_name: str) -> types.ModuleType:
    """Load a module by name."""
    loaded = sys.modules.get(module_name)
    if loaded is not None:
        return loaded

    if importlib_util.find_spec(module_name) is not None:
        return importlib.import_module(module_name)

    msg = f"couldn't load module for '{module_name}'"
    raise ImportError(msg)


def optional_module(
    module_name: str, *, package: str | None = None, extra: str = "3d", message: str | None = None
) -> types.ModuleType | MissingOptionalDependency:
    """Load an optional module or return a raising proxy.

    Args:
        module_name (str): Module to import.
        package (str | None, optional): Package name to show in the error. If None, uses the
            top-level module name. Default: None.
        extra (str, optional): Package extra to show in the installation hint. Default: "3d".
        message (str | None, optional): Custom error message. Default: None.

    Returns:
        types.ModuleType | MissingOptionalDependency: Imported module or a missing-dependency proxy.

    """
    package_name = package or module_name.split(".", maxsplit=1)[0]
    try:
        return load_module(module_name)
    except ImportError:
        return MissingOptionalDependency(package_name, module_name, extra=extra, message=message)


def optional_attr(
    module_name: str, attr_name: str, *, package: str | None = None, extra: str = "3d", message: str | None = None
) -> Any:
    """Load an optional module attribute or return a raising proxy.

    Args:
        module_name (str): Module containing the attribute.
        attr_name (str): Attribute to load from the module.
        package (str | None, optional): Package name to show in the error. If None, uses the
            top-level module name. Default: None.
        extra (str, optional): Package extra to show in the installation hint. Default: "3d".
        message (str | None, optional): Custom error message. Default: None.

    Returns:
        Any: Imported attribute or a missing-dependency proxy.

    """
    package_name = package or module_name.split(".", maxsplit=1)[0]
    try:
        module = load_module(module_name)
        return getattr(module, attr_name)
    except (AttributeError, ImportError):
        return MissingOptionalDependency(package_name, f"{module_name}.{attr_name}", extra=extra, message=message)


def module_from_obj(x: Any) -> types.ModuleType:
    """Get the module for a given object."""
    return load_module(module_name_from_obj(x))


def module_available(module: str, log_warning: str | None = None) -> bool:
    """Check if a module is available.

    Args:
        module (str): The name of the module to check.
        log_warning (str | None): Optional warning message to log if the module is not available.

    Returns:
        bool: True if the module is installed, False otherwise.

    """
    if importlib_util.find_spec(module) is not None:
        return True

    if log_warning is not None:
        logger.warning(log_warning)
    return False


def requires(package: str | list[str], msg: str | None = None, *, extra: str | None = None) -> None:
    """Check if a package or list of packages is available and raise ImportError if not.

    Args:
        package (str | list[str]): The name of the package to check.
        msg (str | None, optional): Optional custom error message to raise if the package is not
            available. Default: None.
        extra (str | None, optional): Optional tinytools extra to show in the installation hint.
            If None, no install hint is added. Default: None.

    Raises:
        ImportError: If the package is not available.

    """
    package = [package] if isinstance(package, str) else package
    available = [module_available(p) for p in package]
    if not all(available):
        missing = [p for p, inst in zip(package, available) if not inst]
        missing_packages = ", ".join(missing)
        parts = []
        if extra is not None:
            parts.append(DEFAULT_OPTIONAL_DEPENDENCY_MESSAGE.format(package=missing_packages, extra=extra))
        else:
            parts.append(f"Required package(s) {missing_packages} not available.")
        if msg is not None and msg.strip() != "":
            parts.append(msg)
        raise ImportError(" ".join(parts))
