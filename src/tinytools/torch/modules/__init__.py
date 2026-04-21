import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .constant_layer import ConstantLayer
    from .ff_block import FFBlock
    from .gated_mlp import GatedMLP
    from .location_head import LocationHead
    from .vanilla_mlp import VanillaMLP

_LAZY_MAPPING = {
    "ConstantLayer": ".constant_layer",
    "FFBlock": ".ff_block",
    "GatedMLP": ".gated_mlp",
    "LocationHead": ".location_head",
    "VanillaMLP": ".vanilla_mlp",
}

__all__ = tuple(sorted(_LAZY_MAPPING))


def __getattr__(name: str) -> Any:
    """Lazily load torch module classes."""
    if name in _LAZY_MAPPING:
        module = importlib.import_module(_LAZY_MAPPING[name], __name__)
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
