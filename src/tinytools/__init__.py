import importlib
from typing import TYPE_CHECKING, Any

from .archives import safe_tar_extract_all, safe_zip_extract_all
from .camera import focal_to_fov, fov_to_focal
from .image import image_grid, img_from_array, imgs_from_array_batch
from .imports import module_available, module_from_obj, requires
from .logger import get_logger, setup_prettier_logger
from .metaclasses import FrozenNamespaceMeta
from .process_annotations import bbox_center, pad_bboxes, process_bboxes, process_seg_masks
from .suppressors import suppress_logging, suppress_output, suppress_tqdm
from .torch import freeze_model
from .tqdm import setup_prettier_tqdm
from .transforms import resize
from .video import get_video_fps, load_video, load_videos, save_video

if TYPE_CHECKING:
    from .threeD import CoordinateConversions, Pose3D, pt3d_to_trimesh, transform_meshes
    from .vlm.litellm import LiteLLMModel
    from .vlm.openai import OpenAIAPIModel
    from .vlm.vllm import VLLMModel

__all__ = [
    "CoordinateConversions",
    "FrozenNamespaceMeta",
    "LiteLLMModel",
    "OpenAIAPIModel",
    "Pose3D",
    "VLLMModel",
    "bbox_center",
    "focal_to_fov",
    "fov_to_focal",
    "freeze_model",
    "get_logger",
    "get_video_fps",
    "image_grid",
    "img_from_array",
    "imgs_from_array_batch",
    "load_video",
    "load_videos",
    "module_available",
    "module_from_obj",
    "pad_bboxes",
    "process_bboxes",
    "process_seg_masks",
    "pt3d_to_trimesh",
    "requires",
    "resize",
    "safe_tar_extract_all",
    "safe_zip_extract_all",
    "save_video",
    "setup_prettier_logger",
    "setup_prettier_tqdm",
    "suppress_logging",
    "suppress_output",
    "suppress_tqdm",
    "transform_meshes",
]

# An internal mapping from the public name to its source module.
_LAZY_MAPPING = {
    "LiteLLMModel": ".vlm.litellm",
    "VLLMModel": ".vlm.vllm",
    "OpenAIAPIModel": ".vlm.openai",
    "pt3d_to_trimesh": ".threeD.mesh_conversions",
    "transform_meshes": ".threeD.transforms",
    "CoordinateConversions": ".threeD.coordinate_conversions",
    "Pose3D": ".threeD.pose3d",
}


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
