import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .images import prepare_images
    from .intrinsics import prepare_intrinsics
    from .pointmaps import prepare_pointmaps

_LAZY_MAPPING = {"prepare_images": ".images", "prepare_intrinsics": ".intrinsics", "prepare_pointmaps": ".pointmaps"}

__all__ = tuple(sorted(_LAZY_MAPPING))


def __getattr__(name: str) -> Any:
    if name in _LAZY_MAPPING:
        module = importlib.import_module(_LAZY_MAPPING[name], __name__)
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
