import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .coordinate_conversions import CoordinateConversions
    from .mesh_conversions import pt3d_to_trimesh, trimesh_to_pt3d
    from .mesh_operations import simplify_mesh
    from .pose3d import Pose3D
    from .pose_target import (
        ApparentSize,
        DisparitySpace,
        Identity,
        LogarithmicDisparitySpace,
        Naive,
        NormalizedSceneScale,
        NormalizedSceneScaleAndTranslation,
        PoseTarget,
        PoseTargetFactory,
        ScaleShiftInvariant,
        ScaleShiftInvariantWTranslationScale,
    )
    from .transforms import (
        DecomposedTransform,
        broadcast_postcompose,
        compose_transform,
        decompose_transform,
        transform_meshes,
    )
    from .utils import get_scale_and_shift

_LAZY_MAPPING = {
    "CoordinateConversions": ".coordinate_conversions",
    "pt3d_to_trimesh": ".mesh_conversions",
    "trimesh_to_pt3d": ".mesh_conversions",
    "Pose3D": ".pose3d",
    "ApparentSize": ".pose_target",
    "DisparitySpace": ".pose_target",
    "Identity": ".pose_target",
    "LogarithmicDisparitySpace": ".pose_target",
    "Naive": ".pose_target",
    "NormalizedSceneScale": ".pose_target",
    "NormalizedSceneScaleAndTranslation": ".pose_target",
    "PoseTarget": ".pose_target",
    "PoseTargetFactory": ".pose_target",
    "ScaleShiftInvariant": ".pose_target",
    "ScaleShiftInvariantWTranslationScale": ".pose_target",
    "DecomposedTransform": ".transforms",
    "broadcast_postcompose": ".transforms",
    "compose_transform": ".transforms",
    "decompose_transform": ".transforms",
    "transform_meshes": ".transforms",
    "get_scale_and_shift": ".utils",
    "simplify_mesh": ".mesh_operations",
}

__all__ = tuple(sorted(_LAZY_MAPPING))


def __getattr__(name: str) -> Any:
    if name in _LAZY_MAPPING:
        module = importlib.import_module(_LAZY_MAPPING[name], __name__)
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
