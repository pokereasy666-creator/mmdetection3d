from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

from torch import Tensor

BASELINE_FREEZE_MODULE_NAMES = (
    'img_backbone',
    'img_neck',
    'view_transform',
    'pts_voxel_encoder',
    'pts_middle_encoder',
    'fusion_layer',
    'pts_backbone',
    'pts_neck',
    'bbox_head',
    'depth_supervisor',
)


@dataclass(frozen=True)
class SensorMeta:
    """Per-sample calibration and augmentation data for feature sampling."""

    lidar2image: Tensor
    camera_intrinsics: Tensor
    camera2lidar: Tensor
    img_aug_matrix: Tensor
    lidar_aug_matrix: Tensor
    image_shapes: Tuple[Any, ...]
    camera_mask: Optional[Tensor] = None


@dataclass(frozen=True)
class BEVGeometry:
    """Static metric geometry shared by BEV feature levels."""

    point_cloud_range: Tuple[float, ...]
    voxel_size: Tuple[float, ...]
    out_size_factor: int
    feature_stride: Tuple[float, float]


@dataclass(frozen=True)
class RefinementFeatureSources:
    """The only feature and geometry inputs visible to instance refinement."""

    raw_img_feats: Tuple[Tensor, ...]
    lidar_bev: Tensor
    fused_bev: Tensor
    sensor_meta: SensorMeta
    bev_geometry: BEVGeometry


@dataclass(frozen=True)
class FeatureBundle:
    """Detector feature taps for optional, mutually independent extensions."""

    raw_img_feats: Tuple[Tensor, ...]
    image_bev: Tensor
    lidar_bev: Tensor
    fused_bev: Tensor
    head_feat: Sequence[Tensor]
    sensor_meta: SensorMeta
    bev_geometry: BEVGeometry
    depth_aux: Optional[Mapping[str, Tensor]] = None

    def refinement_sources(self) -> RefinementFeatureSources:
        """Return shared tensor references without exposing other fields."""
        return RefinementFeatureSources(
            raw_img_feats=self.raw_img_feats,
            lidar_bev=self.lidar_bev,
            fused_bev=self.fused_bev,
            sensor_meta=self.sensor_meta,
            bev_geometry=self.bev_geometry,
        )


@dataclass(frozen=True)
class ProposalPack:
    """Common query-major proposal data for all feature sources.

    ``query_feat_*`` use ``[B, K, D]`` and ``ref_xy`` contains unnormalized
    BEV cell-center coordinates with shape ``[B, K, 2]``.
    """

    query_feat_pre: Tensor
    query_feat_post: Tensor
    ref_xy: Tensor
    scores: Tensor
    class_scores: Tensor
    labels: Tensor
    indices: Tensor
    dense_heatmap: Tensor
    source_type: str
