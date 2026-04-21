import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .modules import ConstantLayer, FFBlock, GatedMLP, LocationHead, VanillaMLP
    from .utils import freeze_module, get_zero_safe_values

_LAZY_MAPPING = {
    "GatedMLP": ".modules.gated_mlp",
    "FFBlock": ".modules.ff_block",
    "ConstantLayer": ".modules.constant_layer",
    "VanillaMLP": ".modules.vanilla_mlp",
    "LocationHead": ".modules.location_head",
    "freeze_module": ".utils",
    "get_zero_safe_values": ".utils",
}

__all__ = tuple(sorted(_LAZY_MAPPING))


def __getattr__(name: str) -> Any:
    """Lazily load torch-backed attributes."""
    if name in _LAZY_MAPPING:
        module = importlib.import_module(_LAZY_MAPPING[name], __name__)
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
