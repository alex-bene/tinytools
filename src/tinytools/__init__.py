import importlib
from typing import TYPE_CHECKING, Any

from .archives import safe_tar_extract_all, safe_zip_extract_all
from .camera import focal_to_fov, fov_to_focal
from .image import image_grid, img_from_array, imgs_from_array_batch, tensor_to_pil
from .imports import (
    MissingOptionalDependency,
    load_module,
    module_available,
    module_from_obj,
    module_name_from_obj,
    optional_attr,
    optional_module,
    requires,
)
from .logger import get_logger, setup_prettier_logger
from .mask_bbox_tools import bboxes_center, pad_bboxes
from .metaclasses import FrozenNamespaceMeta
from .process_annotations import process_bboxes, process_seg_masks
from .suppressors import suppress_logging, suppress_output, suppress_tqdm
from .threeD import _LAZY_MAPPING as _THREED_LAZY_MAPPING
from .torch import _LAZY_MAPPING as _TORCH_LAZY_MAPPING
from .tqdm import setup_prettier_tqdm
from .transforms import resize
from .video import get_video_fps, load_video, load_videos, save_video

if TYPE_CHECKING:
    from .threeD import (
        ApparentSize,
        CoordinateConversions,
        DecomposedTransform,
        DisparitySpace,
        Identity,
        LogarithmicDisparitySpace,
        Naive,
        NormalizedSceneScale,
        NormalizedSceneScaleAndTranslation,
        Pose3D,
        PoseTarget,
        PoseTargetFactory,
        ScaleShiftInvariant,
        ScaleShiftInvariantWTranslationScale,
        broadcast_postcompose,
        compose_transform,
        decompose_transform,
        get_scale_and_shift,
        pt3d_to_trimesh,
        transform_meshes,
    )
    from .torch import ConstantLayer, FFBlock, GatedMLP, LocationHead, VanillaMLP, freeze_module
    from .vlm.openai import OpenAIAPIModel

_EAGER_EXPORTS = [
    "bboxes_center",
    "focal_to_fov",
    "fov_to_focal",
    "FrozenNamespaceMeta",
    "get_logger",
    "get_video_fps",
    "image_grid",
    "img_from_array",
    "imgs_from_array_batch",
    "load_module",
    "MissingOptionalDependency",
    "load_video",
    "load_videos",
    "module_available",
    "module_from_obj",
    "module_name_from_obj",
    "optional_attr",
    "optional_module",
    "pad_bboxes",
    "process_bboxes",
    "process_seg_masks",
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
    "tensor_to_pil",
]

_THREED_LAZY_EXPORTS = {name: f".threeD{module_path}" for name, module_path in _THREED_LAZY_MAPPING.items()}
_TORCH_LAZY_EXPORTS = dict.fromkeys(_TORCH_LAZY_MAPPING, ".torch")

# An internal mapping from the public name to its source module.
_LAZY_MAPPING = {"OpenAIAPIModel": ".vlm.openai", **_THREED_LAZY_EXPORTS, **_TORCH_LAZY_EXPORTS}

__all__ = tuple(sorted([*_EAGER_EXPORTS, *_LAZY_MAPPING]))


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
