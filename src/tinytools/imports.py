"""Utilities for importing modules."""

from __future__ import annotations

import importlib
import sys
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


def module_available(name: str, msg: str | None = None) -> bool:
    """Check if a module is available."""
    try:
        importlib.import_module(name)
    except ImportError:
        if msg is not None:
            logger.warning(msg)
        return False
    else:
        return True
