import importlib
from typing import TYPE_CHECKING, Any

from .archives import safe_tar_extract_all, safe_zip_extract_all
from .image import image_grid, img_from_array, imgs_from_array_batch
from .logger import get_logger, setup_prettier_root_logger
from .process_annotations import process_bboxes, process_seg_masks
from .tqdm import setup_prettier_tqdm
from .video import get_video_fps, load_video, save_video

if TYPE_CHECKING:
    from .litellm import LiteLLMModel
    from .vllm import VLLMModel


# An internal mapping from the public name to its source module.
_LAZY_MAPPING = {"LitLLMModel": ".litellm", "VLLMModel": ".vllm"}


def __getattr__(name: str) -> Any:
    """Lazily load attributes using the _LAZY_MAPPING dictionary."""
    # Check if the requested name is in our lazy-loading map.
    if name in _LAZY_MAPPING:
        # Get the relative module path from the map.
        module_path = _LAZY_MAPPING[name]
        # Import the module.
        module = importlib.import_module(module_path, __name__)
        # Get the attribute (the class) from the imported module.
        attr = getattr(module, name)
        # Cache it in the current module's namespace for future access.
        globals()[name] = attr
        # Return the attribute.
        return attr

    # If the name isn't in our map, it's a genuine error.
    msg = f"module '{__name__}' has no attribute '{name}'"
    raise AttributeError(msg)
