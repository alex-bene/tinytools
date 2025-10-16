import importlib
from typing import TYPE_CHECKING, Any

from .archives import safe_tar_extract_all, safe_zip_extract_all
from .image import image_grid, img_from_array, imgs_from_array_batch
from .imports import module_from_obj
from .logger import get_logger, setup_prettier_root_logger
from .process_annotations import bbox_center, pad_bboxes, process_bboxes, process_seg_masks
from .suppressors import suppress_logging, suppress_output, suppress_tqdm
from .tqdm import setup_prettier_tqdm
from .video import get_video_fps, load_video, load_videos, save_video

if TYPE_CHECKING:
    from .vlm.litellm import LiteLLMModel
    from .vlm.openai import OpenAIAPIModel
    from .vlm.vllm import VLLMModel

__all__ = [
    "LiteLLMModel",
    "OpenAIAPIModel",
    "VLLMModel",
    "bbox_center",
    "get_logger",
    "get_video_fps",
    "image_grid",
    "img_from_array",
    "imgs_from_array_batch",
    "load_video",
    "load_videos",
    "module_from_obj",
    "pad_bboxes",
    "process_bboxes",
    "process_seg_masks",
    "safe_tar_extract_all",
    "safe_zip_extract_all",
    "save_video",
    "setup_prettier_root_logger",
    "setup_prettier_tqdm",
    "suppress_logging",
    "suppress_output",
    "suppress_tqdm",
]

# An internal mapping from the public name to its source module.
_LAZY_MAPPING = {"LiteLLMModel": ".vlm.litellm", "VLLMModel": ".vlm.vllm", "OpenAIAPIModel": ".vlm.openai"}


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
