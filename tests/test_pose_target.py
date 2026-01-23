from __future__ import annotations

import pytest
import torch

from src.tinytools.threeD.pose_target import (
    ApparentSize,
    DisparitySpace,
    Identity,
    InstancePose,
    InvariantPoseTarget,
    LogarithmicDisparitySpace,
    Naive,
    NormalizedSceneScale,
    NormalizedSceneScaleAndTranslation,
    PoseTarget,
    ScaleShiftInvariant,
    ScaleShiftInvariantWTranslationScale,
)


@pytest.fixture
def instance_pose(request: pytest.FixtureRequest) -> InstancePose:
    param = getattr(request, "param", {})
    ndim = param.get("ndim", 5)
    eps = param.get("eps", 1e-5)

    dim_sizes = torch.randint(1, 15, (ndim,)).tolist()
    return InstancePose(
        scale=torch.rand(*dim_sizes, 3) * 15 + eps,  # or torch.randn(*dim_sizes, 1)
        rotation=torch.eye(3).expand(*dim_sizes, 3, 3),
        translation=torch.rand(*dim_sizes, 3) * 15 + eps,
        scene_scale=torch.rand(*dim_sizes, 1) * 15 + eps,
        scene_shift=torch.rand(*dim_sizes, 3) * 15 + eps,
    )


def test_pose_target_defaults() -> None:
    """Test PoseTarget base class instantiation."""
    translation = torch.rand(5, 4, 2, 3) * 15
    pose_target = PoseTarget(translation=translation)
    pose_target_explicit = PoseTarget(
        rotation=None,
        translation=translation,
        scale=torch.ones(5, 4, 2, 1),
        scene_center=torch.zeros(5, 4, 2, 3),
        scene_scale=torch.ones(5, 4, 2, 1),
        translation_scale=torch.ones(5, 4, 2, 1),
    )

    assert pose_target == pose_target_explicit


def test_instance_pose_defaults() -> None:
    """Test PoseTarget base class instantiation."""
    translation = torch.rand(5, 4, 2, 3) * 15
    instance_pose = InstancePose(translation=translation)
    instance_pose_explicit = InstancePose(
        rotation=None,
        translation=translation,
        scale=torch.ones(5, 4, 2, 1),
        scene_shift=torch.zeros(5, 4, 2, 3),
        scene_scale=torch.ones(5, 4, 2, 1),
    )

    assert instance_pose == instance_pose_explicit


def test_invariant_pose_target(instance_pose: InstancePose) -> None:
    """Test various InvariantPoseTarget conversions."""
    invariant_pose_target = InvariantPoseTarget.from_instance_pose(instance_pose)

    assert torch.allclose(invariant_pose_target.s_scene, instance_pose.scene_scale)
    assert torch.allclose(invariant_pose_target.t_scene_center, instance_pose.scene_shift)
    assert torch.allclose(invariant_pose_target.q, instance_pose.rotation)
    t_unit = invariant_pose_target.t_unit
    assert torch.allclose(t_unit.norm(dim=-1), torch.ones_like(t_unit[..., 0]))
    assert torch.allclose(invariant_pose_target.s_rel, invariant_pose_target.s_tilde * invariant_pose_target.t_rel_norm)
    assert torch.allclose(invariant_pose_target.s_rel, instance_pose.scale / instance_pose.scene_scale)
    assert torch.allclose(
        instance_pose.translation,
        invariant_pose_target.t_unit * invariant_pose_target.t_rel_norm * invariant_pose_target.s_scene,
    )

    assert instance_pose == invariant_pose_target.to_instance_pose()


def test_normalized_scene_scale_pose_target(instance_pose: InstancePose) -> None:
    """Test NormalizedSceneScale PoseTarget conversions."""
    invariant_pose_target = InvariantPoseTarget.from_instance_pose(instance_pose)
    pose_target = NormalizedSceneScale.from_instance_pose(instance_pose)

    assert pose_target.__class__.__name__ == "NormalizedSceneScale"
    assert pose_target.pose_target_convention == "NormalizedSceneScale"
    assert torch.allclose(pose_target.scale, invariant_pose_target.s_rel)
    assert torch.allclose(pose_target.rotation, invariant_pose_target.q)
    assert torch.allclose(pose_target.translation, invariant_pose_target.t_unit * invariant_pose_target.t_rel_norm)
    assert torch.allclose(pose_target.translation_scale, torch.ones_like(pose_target.translation_scale))
    assert torch.allclose(pose_target.scene_scale, invariant_pose_target.s_scene)
    assert torch.allclose(pose_target.scene_center, invariant_pose_target.t_scene_center)

    assert instance_pose == pose_target.to_instance_pose()
    assert invariant_pose_target == pose_target.to_invariant()
    assert pose_target == pose_target.from_invariant(invariant_pose_target)


def test_naive_pose_target(instance_pose: InstancePose) -> None:
    """Test Naive PoseTarget conversions."""
    invariant_pose_target = InvariantPoseTarget.from_instance_pose(instance_pose)
    pose_target = Naive.from_instance_pose(instance_pose)

    assert pose_target.__class__.__name__ == "Naive"
    assert pose_target.pose_target_convention == "Naive"
    assert torch.allclose(pose_target.scale, invariant_pose_target.s_scene * invariant_pose_target.s_rel)
    assert torch.allclose(pose_target.rotation, invariant_pose_target.q)
    assert torch.allclose(pose_target.translation, invariant_pose_target.t_unit * invariant_pose_target.t_rel_norm)
    assert torch.allclose(pose_target.translation_scale, torch.ones_like(pose_target.translation_scale))
    assert torch.allclose(pose_target.scene_scale, invariant_pose_target.s_scene)
    assert torch.allclose(pose_target.scene_center, invariant_pose_target.t_scene_center)

    assert instance_pose == pose_target.to_instance_pose()
    assert invariant_pose_target == pose_target.to_invariant()
    assert pose_target == pose_target.from_invariant(invariant_pose_target)


def test_normalized_scene_scale_and_translation_pose_target(instance_pose: InstancePose) -> None:
    """Test NormalizedSceneScaleAndTranslation PoseTarget conversions."""
    invariant_pose_target = InvariantPoseTarget.from_instance_pose(instance_pose)
    pose_target = NormalizedSceneScaleAndTranslation.from_instance_pose(instance_pose)

    assert pose_target.__class__.__name__ == "NormalizedSceneScaleAndTranslation"
    assert pose_target.pose_target_convention == "NormalizedSceneScaleAndTranslation"
    assert torch.allclose(pose_target.scale, invariant_pose_target.s_rel)
    assert torch.allclose(pose_target.rotation, invariant_pose_target.q)
    assert torch.allclose(pose_target.translation, invariant_pose_target.t_unit)
    assert torch.allclose(pose_target.translation_scale, invariant_pose_target.t_rel_norm)
    assert torch.allclose(pose_target.scene_scale, invariant_pose_target.s_scene)
    assert torch.allclose(pose_target.scene_center, invariant_pose_target.t_scene_center)

    assert instance_pose == pose_target.to_instance_pose()
    assert invariant_pose_target == pose_target.to_invariant()
    assert pose_target == pose_target.from_invariant(invariant_pose_target)


def test_apparent_size_pose_target(instance_pose: InstancePose) -> None:
    """Test ApparentSize PoseTarget conversions."""
    invariant_pose_target = InvariantPoseTarget.from_instance_pose(instance_pose)
    pose_target = ApparentSize.from_instance_pose(instance_pose)

    assert pose_target.__class__.__name__ == "ApparentSize"
    assert pose_target.pose_target_convention == "ApparentSize"
    assert torch.allclose(pose_target.scale, invariant_pose_target.s_tilde)
    assert torch.allclose(pose_target.rotation, invariant_pose_target.q)
    assert torch.allclose(pose_target.translation, invariant_pose_target.t_unit)
    assert torch.allclose(pose_target.translation_scale, invariant_pose_target.t_rel_norm)
    assert torch.allclose(pose_target.scene_scale, invariant_pose_target.s_scene)
    assert torch.allclose(pose_target.scene_center, invariant_pose_target.t_scene_center)

    assert instance_pose == pose_target.to_instance_pose()
    assert invariant_pose_target == pose_target.to_invariant()
    assert pose_target == pose_target.from_invariant(invariant_pose_target)


def test_identity_pose_target(instance_pose: InstancePose) -> None:
    """Test Identity PoseTarget conversions."""
    pose_target = Identity.from_instance_pose(instance_pose)

    assert pose_target.__class__.__name__ == "Identity"
    assert pose_target.pose_target_convention == "Identity"
    assert torch.allclose(pose_target.scale, instance_pose.scale)
    assert torch.allclose(pose_target.rotation, instance_pose.rotation)
    assert torch.allclose(pose_target.translation, instance_pose.translation)
    assert torch.allclose(pose_target.translation_scale, torch.ones_like(pose_target.translation_scale))
    assert torch.allclose(pose_target.scene_scale, instance_pose.scene_scale)
    assert torch.allclose(pose_target.scene_center, instance_pose.scene_shift)

    assert instance_pose == pose_target.to_instance_pose()
    invariant_pose_target = InvariantPoseTarget.from_instance_pose(instance_pose)
    assert invariant_pose_target == pose_target.to_invariant()
    assert pose_target == pose_target.from_invariant(invariant_pose_target)


@pytest.mark.parametrize("instance_pose", [{"eps": 1e-1}], indirect=True)
def test_disparity_space_pose_target(instance_pose: InstancePose) -> None:
    """Test DisparitySpace PoseTarget conversions."""
    pose_target = DisparitySpace.from_instance_pose(instance_pose)

    assert pose_target.__class__.__name__ == "DisparitySpace"
    assert pose_target.pose_target_convention == "DisparitySpace"
    assert instance_pose.__eq__(pose_target.to_instance_pose(), atol=1e-4)
    invariant_pose_target = InvariantPoseTarget.from_instance_pose(instance_pose)
    assert invariant_pose_target.__eq__(pose_target.to_invariant(), atol=1e-4)
    assert pose_target.__eq__(pose_target.from_invariant(invariant_pose_target), atol=1e-4)


@pytest.mark.parametrize("instance_pose", [{"eps": 1e-1}], indirect=True)
def test_logarithmic_disparity_space_pose_target(instance_pose: InstancePose) -> None:
    """Test LogarithmicDisparitySpace PoseTarget conversions."""
    pose_target = LogarithmicDisparitySpace.from_instance_pose(instance_pose)

    assert pose_target.__class__.__name__ == "LogarithmicDisparitySpace"
    assert pose_target.pose_target_convention == "LogarithmicDisparitySpace"
    assert instance_pose.__eq__(pose_target.to_instance_pose(), atol=1e-4)
    invariant_pose_target = InvariantPoseTarget.from_instance_pose(instance_pose)
    assert invariant_pose_target.__eq__(pose_target.to_invariant(), atol=1e-4)
    assert pose_target.__eq__(pose_target.from_invariant(invariant_pose_target), atol=1e-4)


@pytest.mark.parametrize("instance_pose", [{"ndim": 1}], indirect=True)
def test_ssi_pose_target(instance_pose: InstancePose) -> None:
    """Test ScaleShiftInvariant PoseTarget conversions."""
    pose_target = ScaleShiftInvariant.from_instance_pose(instance_pose)

    assert pose_target.__class__.__name__ == "ScaleShiftInvariant"
    assert pose_target.pose_target_convention == "ScaleShiftInvariant"
    assert instance_pose.__eq__(pose_target.to_instance_pose(), atol=1e-4)
    invariant_pose_target = InvariantPoseTarget.from_instance_pose(instance_pose)
    assert invariant_pose_target.__eq__(pose_target.to_invariant(), atol=1e-4)
    assert pose_target.__eq__(pose_target.from_invariant(invariant_pose_target), atol=1e-4)


@pytest.mark.parametrize("instance_pose", [{"ndim": 1}], indirect=True)
def test_ssi_w_translation_scale_pose_target(instance_pose: InstancePose) -> None:
    """Test ScaleShiftInvariantWTranslationScale PoseTarget conversions."""
    pose_target = ScaleShiftInvariantWTranslationScale.from_instance_pose(instance_pose)

    assert pose_target.__class__.__name__ == "ScaleShiftInvariantWTranslationScale"
    assert pose_target.pose_target_convention == "ScaleShiftInvariantWTranslationScale"
    assert instance_pose.__eq__(pose_target.to_instance_pose(), atol=1e-4)
    invariant_pose_target = InvariantPoseTarget.from_instance_pose(instance_pose)
    assert invariant_pose_target.__eq__(pose_target.to_invariant(), atol=1e-4)
    assert pose_target.__eq__(pose_target.from_invariant(invariant_pose_target), atol=1e-4)
