"""Pose Target Representations for 3D Object Pose Estimation.

| Convention Name                      | scale   | translation         | translation_scale   | scene_translation_scale |
|--------------------------------------+---------+---------------------+---------------------+-------------------------|
| Identity                             | s       | t                   | 1.0                 | 1.0                     |
| Naive                                | s       | t_rel               | 1.0                 | 1.0                     |
| ApparentSize                         | s_tilde | t_unit              | t_rel_norm          | 1.0                     |
| NormalizedSceneScale                 | s_rel   | t_rel               | 1.0                 | 1.0                     |
| NormalizedSceneScaleAndTranslation   | s_rel   | t_unit              | t_rel_norm          | 1.0                     |
| ScaleShiftInvariant                  | s / Ss  | (t - Ts) / Ss       | 1.0                 | 1.0                     |
| ScaleShiftInvariantWTranslationScale | s / Ss  | unit((t - Ts) / Ss) | norm((t - Ts) / Ss) | 1.0                     |
| DisparitySpace                       | s / Ss  | (t / Z) - (Ts / Zs) | (Z - Zs) / Zs       | 1 / Zs                  |
| LogarithmicDisparitySpace            | s / Ss  | (t / Z) - (Ts / Zs) | log(Z) - log(Zs)    | log(Zs)                 |

Adapted from SAM3D Object
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from tinytools.imports import requires
from tinytools.logger import get_logger

from .transforms import compose_transform, decompose_transform

try:
    from pytorch3d.transforms import Transform3d as pt3d_Transform3d  # pyright: ignore[reportMissingImports]
except ImportError:
    pt3d_Transform3d = None  # type: ignore[assignment]  # noqa: N816

if TYPE_CHECKING:
    from pytorch3d.transforms import Transform3d  # pyright: ignore[reportMissingImports]

logger = get_logger(__name__)


def get_scale_and_shift(pointmap: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute scale and shift for a pointmap.

    Args:
        pointmap: A tensor of shape (..., 3) representing 3D points.

    Returns:
        A tuple containing:
            - scale: A tensor of shape (3,) representing the scale along each axis.
            - shift: A tensor of shape (3,) representing the shift along each axis.

    """
    shift_z = pointmap[..., -1].nanmedian().unsqueeze(0)
    shift = torch.zeros_like(shift_z.expand(1, 3))
    shift[..., -1] = shift_z

    shifted_pointmap = pointmap - shift
    scale = shifted_pointmap.abs().nanmean().to(shift.device)

    shift = shift.reshape(3)
    scale = scale.expand(3)

    return scale, shift


def ssi_to_metric(scale: torch.Tensor, shift: torch.Tensor) -> Transform3d:
    """Get the Transform3d that converts Scale-Shift-Invariant coordinates to Metric coordinates."""
    requires("pytorch3d", "-> ssi_to_metric")
    if scale.ndim == 1:
        scale = scale.unsqueeze(0)
    if scale.shape[-1] == 1:
        scale = scale.expand(*scale.shape[:-1], 3)
    if shift.ndim == 1:
        shift = shift.unsqueeze(0)
    return pt3d_Transform3d().scale(scale).translate(shift).to(shift.device)


@dataclass
class InstancePose:  # noqa: PLW1641
    """Stores the **Metric** pose of an object in Camera coordinates (Local-to-Camera).

    This class holds the absolute, physical values (e.g. in meters).
    It also stores the `scene_scale` and `scene_shift` as metadata which are required to convert this metrics pose into
    the normalized/relative pose used for training.

    Background:
    ---
    The transformation taking object points to metric camera points (l2c) is defined as
        T(x) = s · R(q) · x + t
        where:
            - x is a point in the object coordinate frame,
            - q is a unit quaternion representing rotation (rotation),
            - s is the object-to-camera scale (scale), and
            - t is the translation (translation).
    The scene parameters are used to normalize points in the camera frame for training as:
        T_norm(x) = ( T(x) - t_scene ) / s_scene
        where:
            - x is a point in the object coordinate frame,
            - s_scene is the scene normalization scale (scene_scale), and
            - t_scene is the scene normalization shift (scene_shift).

    Attributes:
        scale: The object-to-camera scale (Metric).
        rotation: The object-to-camera rotation.
        translation: The object-to-camera translation (Metric).
        scene_scale: The scalar used to normalize the scene size (s_scene).
        scene_shift: The vector used to center the scene (t_scene).

    """

    translation: torch.Tensor
    scale: torch.Tensor | None = None
    rotation: torch.Tensor | None = None
    scene_scale: torch.Tensor | None = None
    scene_shift: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """Complete missing fields with default values."""
        if self.scale is None:
            self.scale = torch.ones_like(self.translation[..., :1])
        if self.rotation is None:
            self.rotation = torch.eye(3, device=self.translation.device).expand(*self.translation.shape[:-1], 3, 3)
        if self.scene_scale is None:
            self.scene_scale = torch.ones_like(self.translation[..., :1])
        if self.scene_shift is None:
            self.scene_shift = torch.zeros_like(self.translation)

    def __eq__(self, other: object, rtol: float = 1e-5, atol: float = 1e-5) -> bool:
        """Check equality between two PoseTarget instances."""
        if not isinstance(other, InstancePose):
            return NotImplemented
        return (
            torch.allclose(self.scale, other.scale, rtol=rtol, atol=atol)
            and torch.allclose(self.rotation, other.rotation, rtol=rtol, atol=atol)
            and torch.allclose(self.translation, other.translation, rtol=rtol, atol=atol)
            and torch.allclose(self.scene_scale, other.scene_scale, rtol=rtol, atol=atol)
            and torch.allclose(self.scene_shift, other.scene_shift, rtol=rtol, atol=atol)
        )

    @classmethod
    def _broadcast_postcompose(
        cls,
        scale: torch.Tensor,
        rotation: torch.Tensor,
        translation: torch.Tensor,
        transform_to_postcompose: Transform3d,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Broadcasted post-composition of a transform defined by scale, rotation, translation with another transform.

        Args:
            scale(torch.Tensor): (B, ..., 1 or 3) tensor of scale factors
            rotation(torch.Tensor): (B, ..., 3, 3) tensor of rotation matrices
            translation(torch.Tensor): (B, ..., 3) tensor of translation vectors
            transform_to_postcompose(Transform3d): transform to post-compose with. Should have shape (B, 4, 4)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Decomposed scale, rotation, translation after
                post-composition.

        """
        single_scale = scale.shape[-1] == 1
        if single_scale:
            scale = scale.expand(*scale.shape[:-1], 3)

        b = scale.shape[0]
        lead_dims = scale.shape[:-1]
        flattened_lead_dims_size = int(torch.prod(torch.tensor(lead_dims)).item())
        # Create transform of shape (flattened_lead_dims_size)
        composed = compose_transform(
            scale=scale.reshape(flattened_lead_dims_size, 3),
            rotation=rotation.reshape(flattened_lead_dims_size, 3, 3),
            translation=translation.reshape(flattened_lead_dims_size, 3),
        )

        # Apply transform to shape (flattened_lead_dims_size)
        pc_transform: torch.Tensor = transform_to_postcompose.get_matrix()  # size B, 4, 4
        pc_transform = pc_transform.repeat(flattened_lead_dims_size // b, 1, 1)  # size (B * K, 4, 4)
        stacked_pc_transform = pt3d_Transform3d(matrix=pc_transform)
        postcomposed = composed.compose(stacked_pc_transform)

        # Decompose back to shape (B, ..., C)
        scale, rotation, translation = decompose_transform(postcomposed)
        scale = scale.reshape(*lead_dims, 3)
        rotation = rotation.reshape(*lead_dims, 3, 3)
        translation = translation.reshape(*lead_dims, 3)
        if single_scale:
            scale = scale[..., 0].unsqueeze(-1)
        return scale, rotation, translation


@dataclass
class InvariantPoseTarget:  # noqa: PLW1641
    """The canonical representation of pose targets, used for computing metrics.

    instance_pose <-> invariant_pose_targets <-> all other pose_target_conventions

    Background:
    ---
    We want to estimate a transformation T: R³ → R³ despite scene scale ambiguity.

    The transformation taking object points to scene points is defined as
        T(x) = s · R(q) · x + t
        where:
            - x is a point in the object coordinate frame,
            - q is a unit quaternion representing rotation,
            - s is the object-to-scene scale, and
            - t is the translation.

    However, there is an inherent scale ambiguity in the scene, denoted as s_scene;
    This ambiguity introduces irreducible error that complicates both evaluation and training.

    To decouple the scene scale from the invariant quantities, we define:
        T(x)  = s_scene · |t_rel| [ s_tilde · R(q) · x + t_unit ]
        where we define
            t_rel = t / s_scene
            s_rel = s / s_scene
            s_tilde = s_rel / |t_rel|
            t_unit = t_rel / |t_rel|

    During training, you would predict (q, s_tilde, t_unit), leaving s_scene separate.

    Hand-wavy error analysis:
    ---
    1. Naive (coupled) estimate:
       T(x) = s_scene [ s_rel · R(q) · x + t_rel ]

       We can define:
           U = ln(s_rel)
           V = ln(|t_rel|)
       so that the error is governed by Var(U + V).

    2. In the decoupled case, we have:
       T(x) = s_scene · |t_rel| [ s_tilde · R(q) · x + t_unit ]
            = s_scene · |t_rel| [ (s_rel / |t_rel|) R(q) · x + t_unit ]
       Then ln(s_tilde) = ln(s_rel) - ln(|t_rel|) = U - V, and the error is
       Var(U - V) = Var(U) + Var(V) - 2Cov(U, V).

    """

    # These are invariant
    q: torch.Tensor
    t_unit: torch.Tensor
    s_scene: torch.Tensor
    t_scene_center: torch.Tensor | None = None
    t_rel_norm: torch.Tensor | None = None
    s_tilde: torch.Tensor | None = None
    s_rel: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """Validate and complete the fields of the InvariantPoseTarget."""
        # Check that fields that are required always have values.
        for field_name in ["q", "t_unit", "s_scene"]:
            if getattr(self, field_name) is None:
                msg = f"Field '{field_name}' must be provided."
                raise ValueError(msg)
        if self.s_rel is None and self.s_tilde is None:
            msg = "Field 's_rel' or 's_tilde' must be provided."
            raise ValueError(msg)
        if self.t_rel_norm is None and (self.s_rel is None or self.s_tilde is None):
            msg = "Field 't_rel_norm' must be provided if one of 's_rel' or 's_tilde' is missing."
            raise ValueError(msg)
        # Infer missing fields.
        if self.s_rel is None:
            self.s_rel = self.s_tilde * self.t_rel_norm
        if self.s_tilde is None:
            self.s_tilde = self.s_rel / self.t_rel_norm
        if self.t_rel_norm is None:
            self.t_rel_norm = self.s_rel / self.s_tilde
        if self.t_scene_center is None:
            self.t_scene_center = torch.zeros_like(self.t_unit)

        # If both are provided, we check for consistency.
        if self.s_tilde is not None and self.t_rel_norm is not None:
            computed_s_tilde = self.s_rel / self.t_rel_norm
            # If the provided s_tilde deviates from what is computed, update it.
            if not torch.allclose(self.s_tilde, computed_s_tilde, atol=1e-6):
                logger.warning(
                    "s_tilde and t_rel_norm are provided, but they are not consistent. Updating s_tilde to %s.",
                    computed_s_tilde,
                )
                self.s_tilde = computed_s_tilde

    @staticmethod
    def from_instance_pose(instance_pose: InstancePose) -> InvariantPoseTarget:
        """Create an InvariantPoseTarget from an InstancePose."""
        q = instance_pose.rotation  # (..., 4)
        s_obj_to_scene = instance_pose.scale  # (..., 1) or (..., 3)
        t_obj_to_scene = instance_pose.translation  # (..., 3)
        s_scene = instance_pose.scene_scale  # (..., 1)
        t_scene_center = instance_pose.scene_shift  # (..., 3)

        s_rel = s_obj_to_scene / s_scene
        t_rel = t_obj_to_scene / s_scene

        # Robust norms
        eps = 1e-8
        t_rel_norm = t_rel.norm(dim=-1, keepdim=True).clamp_min(eps)
        s_tilde = s_rel / t_rel_norm
        t_unit = t_rel / t_rel_norm

        return InvariantPoseTarget(
            q=q,
            s_scene=s_scene,
            t_scene_center=t_scene_center,
            s_rel=s_rel,
            s_tilde=s_tilde,
            t_unit=t_unit,
            t_rel_norm=t_rel_norm,
        )

    def to_instance_pose(self) -> InstancePose:
        """Convert InvariantPoseTarget to InstancePose."""
        # scale factor per the derivation: s_scene * |t_rel|
        # Normalize to scene scale (per the derivation)
        scale = self.s_scene * self.t_rel_norm
        return InstancePose(
            scale=self.s_tilde * scale,
            translation=self.t_unit * scale,
            rotation=self.q,
            scene_scale=self.s_scene,
            scene_shift=self.t_scene_center,
        )

    def __eq__(self, other: object, rtol: float = 1e-5, atol: float = 1e-5) -> bool:
        """Check equality between two PoseTarget instances."""
        if not isinstance(other, InvariantPoseTarget):
            return NotImplemented
        return (
            torch.allclose(self.q, other.q, rtol=rtol, atol=atol)
            and torch.allclose(self.t_unit, other.t_unit, rtol=rtol, atol=atol)
            and torch.allclose(self.s_scene, other.s_scene, rtol=rtol, atol=atol)
            and torch.allclose(self.t_scene_center, other.t_scene_center, rtol=rtol, atol=atol)
            and torch.allclose(self.t_rel_norm, other.t_rel_norm, rtol=rtol, atol=atol)
            and torch.allclose(self.s_tilde, other.s_tilde, rtol=rtol, atol=atol)
            and torch.allclose(self.s_rel, other.s_rel, rtol=rtol, atol=atol)
        )


@dataclass
class PoseTarget:  # noqa: PLW1641
    """A generic container for Neural Network inputs/outputs.

    WARNING: The physical meaning of these fields is **Polymorphic**.
    You MUST check `pose_target_convention` to know what the numbers represent.

    Common Interpretations:
    ---
    1. scale:
       - 'Identity': Physical size (meters).
       - 'DisparitySpace': Apparent Size (physical_size / depth).
       - 'ScaleShiftInvariant': Log-normalized relative size.

    2. translation:
       - 'Identity': Physical position (x, y, z).
       - 'DisparitySpace': Screen Coordinates [u, v, 0] (projected).
       - 'NormalizedSceneScale': Unit Direction Vector (x, y, z normalized).

    3. translation_scale:
       - 'Identity': 1.0 (Unused).
       - 'DisparitySpace': Depth (z) or Log-Depth.
       - 'NormalizedSceneScale': Radial Distance (norm of translation).

    4. scene_...:
       - Contextual info usually provided as input to the network, not predicted.

    Attributes:
        scale:                  Shape/Size representation.
        rotation:               Rotation quaternion (usually invariant).
        translation:            Position/Direction representation.
        scene_scale:            Global scene scale factor.
        scene_center:           Global scene shift vector.
        translation_scale:      Depth/Distance magnitude.
        pose_target_convention: The name of the logic class (e.g., "DisparitySpace").

    """

    translation: torch.Tensor
    scale: torch.Tensor | None = None
    rotation: torch.Tensor | None = None
    scene_scale: torch.Tensor | None = None
    scene_center: torch.Tensor | None = None
    scene_translation_scale: torch.Tensor | None = None
    translation_scale: torch.Tensor | None = None
    pose_target_convention: str = field(default="unknown")

    """Convert between pose_target <-> instance_pose <-> invariant_pose_target."""

    def __post_init__(self) -> None:
        """Complete missing fields with default values."""
        if self.scale is None:
            self.scale = torch.ones_like(self.translation[..., :1])
        if self.rotation is None:
            self.rotation = torch.eye(3, device=self.translation.device).expand(*self.translation.shape[:-1], 3, 3)
        if self.scene_scale is None:
            self.scene_scale = torch.ones_like(self.translation[..., :1])
        if self.scene_center is None:
            self.scene_center = torch.zeros_like(self.translation)
        if self.translation_scale is None:
            self.translation_scale = torch.ones_like(self.translation[..., :1])
        if self.scene_translation_scale is None:
            self.scene_translation_scale = torch.ones_like(self.translation[..., :1])

    @classmethod
    def from_invariant(cls, _: InvariantPoseTarget) -> PoseTarget:
        """Convert InvariantPoseTarget to PoseTarget."""
        msg = "Implement this in a subclass"
        raise NotImplementedError(msg)

    def to_invariant(self) -> InvariantPoseTarget:
        """Convert PoseTarget to InvariantPoseTarget."""
        msg = "Implement this in a subclass"
        raise NotImplementedError(msg)

    @classmethod
    def from_instance_pose(cls, instance_pose: InstancePose) -> PoseTarget:
        """Convert InstancePose to PoseTarget."""
        invariant_targets = InvariantPoseTarget.from_instance_pose(instance_pose)
        return cls.from_invariant(invariant_targets)

    def to_instance_pose(self) -> InstancePose:
        """Convert PoseTarget to InstancePose."""
        invariant_targets = self.to_invariant()
        return invariant_targets.to_instance_pose()

    def __eq__(self, other: object, rtol: float = 1e-5, atol: float = 1e-5) -> bool:
        """Check equality between two PoseTarget instances."""
        if not isinstance(other, PoseTarget):
            return NotImplemented
        return (
            torch.allclose(self.scale, other.scale, rtol=rtol, atol=atol)
            and torch.allclose(self.rotation, other.rotation, rtol=rtol, atol=atol)
            and torch.allclose(self.translation, other.translation, rtol=rtol, atol=atol)
            and torch.allclose(self.scene_scale, other.scene_scale, rtol=rtol, atol=atol)
            and torch.allclose(self.scene_center, other.scene_center, rtol=rtol, atol=atol)
            and torch.allclose(self.translation_scale, other.translation_scale, rtol=rtol, atol=atol)
            and torch.allclose(self.scene_translation_scale, other.scene_translation_scale, rtol=rtol, atol=atol)
            and self.pose_target_convention == other.pose_target_convention
        )


class ScaleShiftInvariant(PoseTarget):
    """Scale and Shift Invariant Pose Target Representation.

    Midas eq. (6): https://arxiv.org/pdf/1907.01341v3
    But for pointmaps (see MoGe): https://arxiv.org/pdf/2410.19115
    """

    pose_target_convention: str = "ScaleShiftInvariant"
    scale_mean = torch.tensor([1.0232692956924438, 1.0232691764831543, 1.0232692956924438]).to(torch.float32)
    scale_std = torch.tensor([1.3773751258850098, 1.3773752450942993, 1.3773750066757202]).to(torch.float32)
    translation_mean = torch.tensor([0.003191213821992278, 0.017236359417438507, 0.9401122331619263]).to(torch.float32)
    translation_std = torch.tensor([1.341888666152954, 0.7665449380874634, 3.175130605697632]).to(torch.float32)

    def __post_init__(self) -> None:
        """Ensure pytorch3d is available."""
        requires("pytorch3d", "-> ScaleShiftInvariant")
        super().__post_init__()

    @classmethod
    def from_instance_pose(cls, instance_pose: InstancePose, normalize: bool = False) -> ScaleShiftInvariant:
        """Convert InstancePose to PoseTarget.

        Basically, it subtracts `scene_shift` and divides by `scene_scale` to get SSI values. Optionally it also
        normalizes scale and translation using pre-computed mean/std.
        """
        scene_scale = instance_pose.scene_scale
        scene_shift = instance_pose.scene_shift

        ssi_scale, ssi_rotation, ssi_translation = InstancePose._broadcast_postcompose(
            scale=instance_pose.scale,
            rotation=instance_pose.rotation,
            translation=instance_pose.translation,
            transform_to_postcompose=ssi_to_metric(scene_scale, scene_shift).inverse(),
        )
        if normalize:
            device = ssi_scale.device
            ssi_scale = (ssi_scale - cls.scale_mean.to(device)) / cls.scale_std.to(device)
            ssi_translation = (ssi_translation - cls.translation_mean.to(device)) / cls.translation_std.to(device)

        return ScaleShiftInvariant(
            scale=ssi_scale,
            rotation=ssi_rotation,
            translation=ssi_translation,
            scene_scale=scene_scale,
            scene_center=scene_shift,
            pose_target_convention=cls.pose_target_convention,
        )

    def to_instance_pose(self, normalize: bool = False) -> InstancePose:
        """Convert PoseTarget to InstancePose.

        Basically, it multiplies by `scene_scale` and adds `scene_shift` to get Metric values. Optionally it also
        denormalizes scale and translation using pre-computed mean/std.
        """
        if normalize:
            # Denormalize
            device = self.scale.device
            self.scale = self.scale * self.scale_std.to(device) + self.scale_mean.to(device)
            self.translation = self.translation * self.translation_std.to(device) + self.translation_mean.to(device)

        ins_scale, ins_rotation, ins_translation = InstancePose._broadcast_postcompose(
            scale=self.scale,
            rotation=self.rotation,
            translation=self.translation,
            transform_to_postcompose=self.ssi_to_metric(),
        )

        return InstancePose(
            scale=ins_scale,
            translation=ins_translation,
            rotation=ins_rotation,
            scene_scale=self.scene_scale,
            scene_shift=self.scene_center,
        )

    def to_invariant(self, normalize: bool = False) -> InvariantPoseTarget:
        """Convert PoseTarget to InvariantPoseTarget."""
        instance_pose = self.to_instance_pose(normalize=normalize)
        return InvariantPoseTarget.from_instance_pose(instance_pose)

    @classmethod
    def from_invariant(cls, invariant_target: InvariantPoseTarget, normalize: bool = False) -> NormalizedSceneScale:
        """Convert InvariantPoseTarget to PoseTarget."""
        instance_pose = invariant_target.to_instance_pose()
        return cls.from_instance_pose(instance_pose, normalize=normalize)

    def ssi_to_metric(self) -> Transform3d:
        """Get the Transform3d that converts Scale-Shift-Invariant coordinates to Metric coordinates."""
        scale = self.scene_scale
        shift = self.scene_center
        return ssi_to_metric(scale, shift)


class ScaleShiftInvariantWTranslationScale(ScaleShiftInvariant):
    """Scale and Shift Invariant Pose Target Representation with translation scale.

    Midas eq. (6): https://arxiv.org/pdf/1907.01341v3
    But for pointmaps (see MoGe): https://arxiv.org/pdf/2410.19115
    """

    pose_target_convention: str = "ScaleShiftInvariantWTranslationScale"

    @classmethod
    def from_instance_pose(
        cls, instance_pose: InstancePose, normalize: bool = False
    ) -> ScaleShiftInvariantWTranslationScale:
        """Convert InstancePose to PoseTarget."""
        ssi_pose_target = super().from_instance_pose(instance_pose, normalize=normalize)
        ssi_pose_target.translation_scale = ssi_pose_target.translation.norm(dim=-1, keepdim=True)
        ssi_pose_target.translation = ssi_pose_target.translation / ssi_pose_target.translation_scale.clamp_min(1e-7)
        ssi_pose_target.pose_target_convention = cls.pose_target_convention
        ssi_pose_target.__class__ = cls
        return ssi_pose_target

    def to_instance_pose(self, normalize: bool = False) -> InstancePose:
        """Convert PoseTarget to InstancePose.

        We could have reused the parent class method, but we would have needed to either modify the input pose_target
        in-place (not good) or make a copy (less efficient). So we just reimplement it here.
        """
        ins_translation_unit = self.translation / self.translation.norm(
            dim=-1, keepdim=True
        )  # it should already be unit, but just in case
        ins_translation = ins_translation_unit * self.translation_scale

        if normalize:
            # Denormalize
            device = self.scale.device
            self.scale = self.scale * self.scale_std.to(device) + self.scale_mean.to(device)
            ins_translation = ins_translation * self.translation_std.to(device) + self.translation_mean.to(device)

        ins_scale, ins_rotation, ins_translation = InstancePose._broadcast_postcompose(
            scale=self.scale,
            rotation=self.rotation,
            translation=ins_translation,
            transform_to_postcompose=self.ssi_to_metric(),
        )

        return InstancePose(
            scale=ins_scale,
            translation=ins_translation,
            rotation=ins_rotation,
            scene_scale=self.scene_scale,
            scene_shift=self.scene_center,
        )


class DisparitySpace(PoseTarget):
    """Disparity Space Pose Target Representation.

    Basically, x and y are divided by depth (z) and depth is stored as 1/z.
    For this representation we use `scene_translation_scale` to store 1/Zs such that
    `scene_translation_scale * scene_shift` gets as the actual scene center. Similarly, `translation_scale` is
    1/depth such that `translation_scale` gives the actual depth and `translation_scale * translation`
    gives the actual object xyz.
    """

    pose_target_convention: str = "DisparitySpace"
    eps: float = 1e-6

    @classmethod
    def from_instance_pose(cls, instance_pose: InstancePose) -> DisparitySpace:
        """Convert InstancePose to PoseTarget."""
        # `scene_center`: [Xs/Zs, Ys/Zs, 1]
        # `scene_translation_scale`: 1/Zs
        # `scene_scale`: scene_scale
        # `translation`: [X/Z, Y/Z, 1] - [Xs/Zs, Ys/Zs, 1]
        # `translation_scale`: (Z - Zs) / Zs
        # `scale`: orig_scale / scene_scale

        # To uv space (i.e., divide by z) with log depth
        scene_shift = instance_pose.scene_shift
        scene_shift_z = scene_shift[..., -1:]
        scene_shift = scene_shift / scene_shift_z  # [Xs/Zs, Ys/Zs, 1]
        scene_translation_scale = torch.clamp_min(scene_shift_z, min=cls.eps).reciprocal()  # 1/Zs

        pose_translation = instance_pose.translation
        pose_z = pose_translation[..., -1:]
        pose_translation = pose_translation / pose_z  # [X/Z, Y/Z, 1]
        pose_translation_scale = torch.clamp_min(pose_z, min=cls.eps).reciprocal()  # 1/Z

        # Compute relative translation and scales
        pose_translation = pose_translation - scene_shift  # [X/Z - Xs/Zs, Y/Z - Ys/Zs, 0]
        pose_translation_scale = (pose_z - scene_shift_z) / scene_shift_z  # (Z - Zs) / Zs

        return DisparitySpace(
            scale=instance_pose.scale / instance_pose.scene_scale,  # orig_scale / scene_scale
            translation=pose_translation,  # [X/Z - Xs/Zs, Y/Z - Ys/Zs, 0]
            translation_scale=pose_translation_scale,  # (Z - Zs) / Zs
            rotation=instance_pose.rotation,
            scene_center=scene_shift,  # [Xs/Zs, Ys/Zs, 1]
            scene_scale=instance_pose.scene_scale,
            scene_translation_scale=scene_translation_scale,  # 1/Zs
            pose_target_convention=cls.pose_target_convention,
        )

    def to_instance_pose(self) -> InstancePose:
        """Convert PoseTarget to InstancePose."""
        # Recover actual scene shift
        scene_shift_z = torch.clamp_min(self.scene_translation_scale, min=self.eps).reciprocal()  # Zs
        scene_shift = self.scene_center * scene_shift_z  # [Xs, Ys, Zs]
        # Recover instance translation
        ins_translation = self.translation.clone()  # [X/Z - Xs/Zs, Y/Z - Ys/Zs, 0]
        ins_translation += self.scene_center  # [X/Z, Y/Z, 1]
        ins_translation_z = (self.translation_scale * scene_shift_z) + scene_shift_z  # Z
        ins_translation *= ins_translation_z  # [X, Y, Z]

        return InstancePose(
            scale=self.scale * self.scene_scale,
            translation=ins_translation,
            rotation=self.rotation,
            scene_scale=self.scene_scale,
            scene_shift=scene_shift,
        )

    def to_invariant(self) -> InvariantPoseTarget:
        """Convert PoseTarget to InvariantPoseTarget."""
        instance_pose = self.to_instance_pose()
        return InvariantPoseTarget.from_instance_pose(instance_pose)

    @classmethod
    def from_invariant(cls, invariant_target: InvariantPoseTarget) -> DisparitySpace:
        """Convert InvariantPoseTarget to PoseTarget."""
        instance_pose = invariant_target.to_instance_pose()
        return cls.from_instance_pose(instance_pose)


class LogarithmicDisparitySpace(PoseTarget):
    """Logarithmic Disparity Space Pose Target Representation.

    Basically, x and y are divided by depth (z) and depth is stored as log(z). Note that `log(1/z) = -log(z)` so this is
    more or less the same as storing 1/z but log(z) is easier for NN to compute as any prediction is valid once we apply
    the exponential to get the actual depth.
    For this representation we use `scene_translation_scale` to store log(Zs) such that
    `exp(scene_translation_scale) * scene_shift` gets as the actual scene center. Similarly, `translation_scale` is
    log(depth) such that `exp(translation_scale)` gives the actual depth and `exp(translation_scale) * translation`
    gives the actual object xyz.
    """

    pose_target_convention: str = "LogarithmicDisparitySpace"
    eps: float = 1e-6

    @classmethod
    def from_instance_pose(cls, instance_pose: InstancePose) -> LogarithmicDisparitySpace:
        """Convert InstancePose to PoseTarget."""
        # `scene_center`: [Xs/Zs, Ys/Zs, 1]
        # `scene_translation_scale`: log(Zs)
        # `scene_scale`: scene_scale
        # `translation`: [X/Z, Y/Z, 1] - [Xs/Zs, Ys/Zs, 1]
        # `translation_scale`: log(Z) - log(Zs) = log(Z / Zs)
        # `scale`: orig_scale / scene_scale

        # To uv space (i.e., divide by z) with log depth
        scene_shift = instance_pose.scene_shift
        scene_shift_z = scene_shift[..., -1:]
        scene_shift = scene_shift / torch.clamp_min(scene_shift_z, min=cls.eps)  # [Xs/Zs, Ys/Zs, 1]
        scene_shift[..., -1] = 1.0  # enforce one z-component
        scene_translation_scale = torch.log(torch.clamp_min(scene_shift_z, min=cls.eps))  # log(Zs)

        pose_translation = instance_pose.translation
        pose_z = pose_translation[..., -1:]
        pose_translation = pose_translation / torch.clamp_min(pose_z, min=cls.eps)  # [X/Z, Y/Z, 1]
        pose_translation[..., -1] = 1.0  # enforce one z-component
        pose_translation_scale = torch.log(torch.clamp_min(pose_z, min=cls.eps))  # log(Z)

        # Compute relative translation and scales
        pose_translation = pose_translation - scene_shift  # [X/Z - Xs/Zs, Y/Z - Ys/Zs, 0]
        pose_translation[..., -1] = 0.0  # enforce zero z-component
        pose_translation_scale -= scene_translation_scale  # log(Z / Zs)

        return LogarithmicDisparitySpace(
            scale=instance_pose.scale / torch.clamp_min(instance_pose.scene_scale, min=cls.eps),  # orig_scale / Ss
            translation=pose_translation,  # [X/Z - Xs/Zs, Y/Z - Ys/Zs, 0]
            translation_scale=pose_translation_scale,  # log(Z / Zs)
            rotation=instance_pose.rotation,
            scene_center=scene_shift,  # [Xs/Zs, Ys/Zs, 1]
            scene_scale=instance_pose.scene_scale,
            scene_translation_scale=scene_translation_scale,  # log(Zs)
            pose_target_convention=cls.pose_target_convention,
        )

    def to_instance_pose(self) -> InstancePose:
        """Convert PoseTarget to InstancePose."""
        # Recover instance translation
        ins_translation = self.translation.clone()  # [X/Z - Xs/Zs, Y/Z - Ys/Zs, 0]
        ins_translation += self.scene_center  # [X/Z, Y/Z, 1]
        ins_translation[..., -1] = 1.0  # enforce one z-component
        ins_translation_z = torch.exp(self.translation_scale + self.scene_translation_scale)  # Z
        ins_translation *= ins_translation_z  # [X, Y, Z]
        # Recover actual scene shift
        scene_shift_z = torch.exp(self.scene_translation_scale)  # Zs
        scene_shift = self.scene_center * scene_shift_z  # [Xs, Ys, Zs]

        return InstancePose(
            scale=self.scale * self.scene_scale,
            translation=ins_translation,
            rotation=self.rotation,
            scene_scale=self.scene_scale,
            scene_shift=scene_shift,
        )

    def to_invariant(self) -> InvariantPoseTarget:
        """Convert PoseTarget to InvariantPoseTarget."""
        instance_pose = self.to_instance_pose()
        return InvariantPoseTarget.from_instance_pose(instance_pose)

    @classmethod
    def from_invariant(cls, invariant_target: InvariantPoseTarget) -> LogarithmicDisparitySpace:
        """Convert InvariantPoseTarget to PoseTarget."""
        instance_pose = invariant_target.to_instance_pose()
        return cls.from_instance_pose(instance_pose)


class NormalizedSceneScale(PoseTarget):
    """NormalizedSceneScale Pose Target Representation.

    This representation normalizes (divides) the scale and translation of the object instance by scene_scale.
    So, for PoseTarget scale/rotation/translation the model would predict s_rel/q/t_rel from InvariantPoseTarget.
    """

    pose_target_convention: str = "NormalizedSceneScale"

    @classmethod
    def from_invariant(cls, invariant_targets: InvariantPoseTarget) -> NormalizedSceneScale:
        """Convert InvariantPoseTarget to PoseTarget."""
        translation = invariant_targets.t_unit * invariant_targets.t_rel_norm
        return NormalizedSceneScale(
            scale=invariant_targets.s_rel,
            rotation=invariant_targets.q,
            translation=translation,
            scene_scale=invariant_targets.s_scene,
            scene_center=invariant_targets.t_scene_center,
            pose_target_convention=cls.pose_target_convention,
        )

    def to_invariant(self) -> InvariantPoseTarget:
        """Convert PoseTarget to InvariantPoseTarget."""
        t_rel_norm = torch.norm(self.translation, dim=-1, keepdim=True)
        return InvariantPoseTarget(
            s_scene=self.scene_scale,
            s_rel=self.scale,
            q=self.rotation,
            t_unit=self.translation / t_rel_norm,
            t_rel_norm=t_rel_norm,
            t_scene_center=self.scene_center,
        )


class Naive(PoseTarget):
    """Naive Pose Target Representation.

    This representation uses s (total) and t_rel as target quantities for PoseTarget.
    """

    pose_target_convention: str = "Naive"

    @classmethod
    def from_invariant(cls, invariant_targets: InvariantPoseTarget) -> Naive:
        """Convert InvariantPoseTarget to PoseTarget."""
        translation = invariant_targets.t_unit * invariant_targets.t_rel_norm
        return Naive(
            scale=invariant_targets.s_rel * invariant_targets.s_scene,
            rotation=invariant_targets.q,
            translation=translation,
            scene_scale=invariant_targets.s_scene,
            scene_center=invariant_targets.t_scene_center,
            pose_target_convention=cls.pose_target_convention,
        )

    def to_invariant(self) -> InvariantPoseTarget:
        """Convert PoseTarget to InvariantPoseTarget."""
        t_rel_norm = torch.norm(self.translation, dim=-1, keepdim=True)
        return InvariantPoseTarget(
            s_scene=self.scene_scale,
            t_scene_center=self.scene_center,
            s_rel=self.scale / self.scene_scale,
            q=self.rotation,
            t_unit=self.translation / t_rel_norm,
            t_rel_norm=t_rel_norm,
        )


class NormalizedSceneScaleAndTranslation(PoseTarget):
    """NormalizedSceneScaleAndTranslation Pose Target Representation.

    This representation uses s_rel, t_unit, and t_rel_norm as target quantities for PoseTarget.
    """

    pose_target_convention: str = "NormalizedSceneScaleAndTranslation"

    @classmethod
    def from_invariant(cls, invariant_targets: InvariantPoseTarget) -> NormalizedSceneScaleAndTranslation:
        """Convert InvariantPoseTarget to PoseTarget."""
        return NormalizedSceneScaleAndTranslation(
            scale=invariant_targets.s_rel,
            rotation=invariant_targets.q,
            translation=invariant_targets.t_unit,
            scene_scale=invariant_targets.s_scene,
            scene_center=invariant_targets.t_scene_center,
            translation_scale=invariant_targets.t_rel_norm,
            pose_target_convention=cls.pose_target_convention,
        )

    def to_invariant(self) -> InvariantPoseTarget:
        """Convert PoseTarget to InvariantPoseTarget."""
        return InvariantPoseTarget(
            s_scene=self.scene_scale,
            t_scene_center=self.scene_center,
            s_rel=self.scale,
            q=self.rotation,
            t_unit=self.translation,
            t_rel_norm=self.translation_scale,
        )


class ApparentSize(PoseTarget):
    """ApparentSize Pose Target Representation.

    This representation uses s_tilde, t_unit and t_rel_norm as target quantities for PoseTarget.
    """

    pose_target_convention: str = "ApparentSize"

    @classmethod
    def from_invariant(cls, invariant_targets: InvariantPoseTarget) -> ApparentSize:
        """Convert InvariantPoseTarget to PoseTarget."""
        return ApparentSize(
            scale=invariant_targets.s_tilde,
            rotation=invariant_targets.q,
            translation=invariant_targets.t_unit,
            scene_scale=invariant_targets.s_scene,
            scene_center=invariant_targets.t_scene_center,
            translation_scale=invariant_targets.t_rel_norm,
            pose_target_convention=cls.pose_target_convention,
        )

    def to_invariant(self) -> InvariantPoseTarget:
        """Convert PoseTarget to InvariantPoseTarget."""
        return InvariantPoseTarget(
            s_scene=self.scene_scale,
            t_scene_center=self.scene_center,
            s_tilde=self.scale,
            q=self.rotation,
            t_unit=self.translation,
            t_rel_norm=self.translation_scale,
        )


class Identity(PoseTarget):
    """Identity Pose Target Representation.

    Basically, direct passthrough mapping between instance pose and pose target values.
    This preserves all values including scene_scale and scene_shift.
    """

    pose_target_convention: str = "Identity"

    @classmethod
    def from_instance_pose(cls, instance_pose: InstancePose) -> Identity:
        """Convert InstancePose to PoseTarget."""
        return Identity(
            scale=instance_pose.scale,
            rotation=instance_pose.rotation,
            translation=instance_pose.translation,
            scene_scale=instance_pose.scene_scale,
            scene_center=instance_pose.scene_shift,
            pose_target_convention=cls.pose_target_convention,
        )

    def to_instance_pose(self) -> InstancePose:
        """Convert PoseTarget to InstancePose."""
        return InstancePose(
            scale=self.scale,
            translation=self.translation,
            rotation=self.rotation,
            scene_scale=self.scene_scale,
            scene_shift=self.scene_center,
        )

    def to_invariant(self) -> InvariantPoseTarget:
        """Convert PoseTarget to InvariantPoseTarget."""
        instance_pose = self.to_instance_pose()
        return InvariantPoseTarget.from_instance_pose(instance_pose)

    @classmethod
    def from_invariant(cls, invariant_target: InvariantPoseTarget) -> Identity:
        """Convert InvariantPoseTarget to PoseTarget."""
        instance_pose = invariant_target.to_instance_pose()
        return cls.from_instance_pose(instance_pose)


PoseTargetFactory: dict[str, type[PoseTarget]] = {
    "ScaleShiftInvariant": ScaleShiftInvariant,
    "ScaleShiftInvariantWTranslationScale": ScaleShiftInvariantWTranslationScale,
    "DisparitySpace": DisparitySpace,
    "LogarithmicDisparitySpace": LogarithmicDisparitySpace,
    "NormalizedSceneScale": NormalizedSceneScale,
    "Naive": Naive,
    "NormalizedSceneScaleAndTranslation": NormalizedSceneScaleAndTranslation,
    "ApparentSize": ApparentSize,
    "Identity": Identity,
}
