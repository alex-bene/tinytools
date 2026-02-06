import importlib
from typing import TYPE_CHECKING, Any

from .archives import safe_tar_extract_all, safe_zip_extract_all
from .camera import focal_to_fov, fov_to_focal
from .image import image_grid, img_from_array, imgs_from_array_batch, tensor_to_pil
from .imports import module_available, module_from_obj, requires
from .logger import get_logger, setup_prettier_logger
from .metaclasses import FrozenNamespaceMeta
from .process_annotations import bbox_center, pad_bboxes, process_bboxes, process_seg_masks
from .suppressors import suppress_logging, suppress_output, suppress_tqdm
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
    from .vlm.vllm import VLLMModel

__all__ = [
    "ApparentSize",
    "ConstantLayer",
    "CoordinateConversions",
    "DecomposedTransform",
    "DisparitySpace",
    "FFBlock",
    "FrozenNamespaceMeta",
    "GatedMLP",
    "Identity",
    "LocationHead",
    "LogarithmicDisparitySpace",
    "Naive",
    "NormalizedSceneScale",
    "NormalizedSceneScaleAndTranslation",
    "OpenAIAPIModel",
    "Pose3D",
    "PoseTarget",
    "PoseTargetFactory",
    "ScaleShiftInvariant",
    "ScaleShiftInvariantWTranslationScale",
    "VLLMModel",
    "VanillaMLP",
    "bbox_center",
    "broadcast_postcompose",
    "compose_transform",
    "decompose_transform",
    "focal_to_fov",
    "fov_to_focal",
    "freeze_module",
    "get_logger",
    "get_scale_and_shift",
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
    "tensor_to_pil",
    "transform_meshes",
]

# An internal mapping from the public name to its source module.
_LAZY_MAPPING = {
    "VLLMModel": ".vlm.vllm",
    "OpenAIAPIModel": ".vlm.openai",
    "pt3d_to_trimesh": ".threeD.mesh_conversions",
    "transform_meshes": ".threeD.transforms",
    "decompose_transform": ".threeD.transforms",
    "compose_transform": ".threeD.transforms",
    "DecomposedTransform": ".threeD.transforms",
    "CoordinateConversions": ".threeD.coordinate_conversions",
    "Pose3D": ".threeD.pose3d",
    "GatedMLP": ".torch.modules.gated_mlp",
    "PreNormBlock": ".torch.modules.prenorm_block",
    "VanillaMLP": ".torch.modules.vanilla_mlp",
    "freeze_module": ".torch.utils",
    "LocationHead": ".torch.modules.location_head",
    "ApparentSize": ".threeD.pose_target",
    "DisparitySpace": ".threeD.pose_target",
    "Identity": ".threeD.pose_target",
    "LogarithmicDisparitySpace": ".threeD.pose_target",
    "Naive": ".threeD.pose_target",
    "NormalizedSceneScale": ".threeD.pose_target",
    "NormalizedSceneScaleAndTranslation": ".threeD.pose_target",
    "PoseTarget": ".threeD.pose_target",
    "PoseTargetFactory": ".threeD.pose_target",
    "ScaleShiftInvariant": ".threeD.pose_target",
    "ScaleShiftInvariantWTranslationScale": ".threeD.pose_target",
    "broadcast_postcompose": ".threeD.transforms",
    "get_scale_and_shift": ".threeD.utils",
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
