# flake8: noqa: E402
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('mmcv')
pytest.importorskip('mmengine')
pytest.importorskip('mmdet')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bevfusion.insfusion import LiDARProposalGenerator  # noqa: E402
from bevfusion.bevfusion import BEVFusion  # noqa: E402
from bevfusion.transfusion_head import TransFusionHead  # noqa: E402
from bevfusion.proposal_utils import (
    build_proposal_pack,  # noqa: E402
    create_2d_grid)
from bevfusion.structures import (
    BEVGeometry,
    FeatureBundle,  # noqa: E402
    ProposalPack,
    SensorMeta)


def _sensor_meta(batch_size, num_cameras):
    eye = torch.eye(4).view(1, 1, 4, 4).repeat(batch_size, num_cameras, 1, 1)
    return SensorMeta(
        lidar2image=eye,
        camera_intrinsics=eye,
        camera2lidar=eye,
        img_aug_matrix=eye,
        lidar_aug_matrix=torch.eye(4).repeat(batch_size, 1, 1),
        image_shapes=tuple((256, 704) for _ in range(batch_size)),
        camera_mask=torch.ones(batch_size, num_cameras, dtype=torch.bool),
    )


def _geometry():
    return BEVGeometry(
        point_cloud_range=(-54., -54., -5., 54., 54., 3.),
        voxel_size=(0.075, 0.075, 0.2),
        out_size_factor=8,
        feature_stride=(0.6, 0.6),
    )


def test_feature_bundle_contract_shares_tensor_objects():
    raw = (torch.randn(2, 6, 8, 4, 5), torch.randn(2, 6, 8, 2, 3))
    image_bev = torch.randn(2, 4, 8, 8)
    lidar_bev = torch.randn(2, 8, 8, 8)
    fused_bev = torch.randn(2, 8, 8, 8)
    head_feat = [torch.randn(2, 16, 8, 8)]
    bundle = FeatureBundle(
        raw_img_feats=raw,
        image_bev=image_bev,
        lidar_bev=lidar_bev,
        fused_bev=fused_bev,
        head_feat=head_feat,
        sensor_meta=_sensor_meta(2, 6),
        bev_geometry=_geometry(),
        depth_aux=None,
    )

    sources = bundle.refinement_sources()
    assert sources.raw_img_feats[0] is raw[0]
    assert sources.lidar_bev is lidar_bev
    assert sources.fused_bev is fused_bev
    assert not isinstance(bundle, torch.nn.Module)
    assert not isinstance(sources, torch.nn.Module)
    assert bundle.depth_aux is None
    assert not hasattr(sources, 'image_bev')
    assert not hasattr(sources, 'head_feat')
    assert not hasattr(sources, 'depth_aux')


def test_optional_depth_aux_stays_outside_refinement_sources():
    tensor = torch.randn(1, 2, 3, 3)
    common = dict(
        raw_img_feats=(tensor, ),
        image_bev=tensor,
        lidar_bev=tensor,
        fused_bev=tensor,
        head_feat=[tensor],
        sensor_meta=_sensor_meta(1, 1),
        bev_geometry=_geometry(),
    )
    without_depth = FeatureBundle(**common)
    with_depth = FeatureBundle(
        **common, depth_aux={'simulated_depth_tensor': tensor})
    lhs = without_depth.refinement_sources()
    rhs = with_depth.refinement_sources()
    assert lhs.raw_img_feats[0] is rhs.raw_img_feats[0]
    assert lhs.lidar_bev is rhs.lidar_bev
    assert lhs.fused_bev is rhs.fused_bev
    assert not hasattr(rhs, 'depth_aux')


def test_dummy_alternative_fuser_satisfies_bundle_contract():

    class DummyAlternativeFuser(torch.nn.Module):

        def forward(self, inputs):
            image_bev, lidar_bev = inputs
            return lidar_bev + image_bev.mean(dim=1, keepdim=True)

    image_bev = torch.randn(1, 4, 8, 8)
    lidar_bev = torch.randn(1, 8, 8, 8)
    fused_bev = DummyAlternativeFuser()([image_bev, lidar_bev])
    bundle = FeatureBundle(
        raw_img_feats=(torch.randn(1, 2, 4, 4, 4), ),
        image_bev=image_bev,
        lidar_bev=lidar_bev,
        fused_bev=fused_bev,
        head_feat=[torch.randn(1, 16, 8, 8)],
        sensor_meta=_sensor_meta(1, 2),
        bev_geometry=_geometry(),
    )
    assert bundle.refinement_sources().fused_bev is fused_bev


def test_detector_extracts_bundle_with_dummy_alternative_fuser():

    class DummyAlternativeFuser(torch.nn.Module):

        def forward(self, inputs):
            self.inputs = inputs
            image_bev, lidar_bev = inputs
            return lidar_bev + image_bev.mean(dim=1, keepdim=True)

    class SingleLevelNeck(torch.nn.Module):

        def forward(self, tensor):
            return [tensor]

    detector = BEVFusion.__new__(BEVFusion)
    torch.nn.Module.__init__(detector)
    detector.instance_refiner = torch.nn.Identity()
    detector.fusion_layer = DummyAlternativeFuser()
    detector.pts_backbone = torch.nn.Identity()
    detector.pts_neck = SingleLevelNeck()
    detector._bev_geometry_cfg = dict(
        point_cloud_range=(-54., -54., -5., 54., 54., 3.),
        voxel_size=(0.075, 0.075, 0.2),
        out_size_factor=8,
    )

    raw = (torch.randn(1, 2, 4, 4, 4), torch.randn(1, 2, 4, 2, 2))
    image_bev = torch.randn(1, 4, 8, 8)
    lidar_bev = torch.randn(1, 8, 8, 8)

    def extract_img_feat(self, *args, **kwargs):
        assert kwargs['return_raw_img_feats']
        return image_bev, raw

    def extract_pts_feat(self, batch_inputs_dict):
        return lidar_bev

    detector.extract_img_feat = types.MethodType(extract_img_feat, detector)
    detector.extract_pts_feat = types.MethodType(extract_pts_feat, detector)
    eye = torch.eye(4).repeat(2, 1, 1).numpy()
    meta = dict(lidar2img=eye, cam2img=eye, cam2lidar=eye)
    bundle = detector.extract_feature_bundle(
        dict(imgs=torch.randn(1, 2, 3, 16, 16), points=[torch.randn(4, 5)]),
        [meta],
    )

    assert detector.fusion_layer.inputs[0] is image_bev
    assert detector.fusion_layer.inputs[1] is lidar_bev
    assert bundle.raw_img_feats[0] is raw[0]
    assert bundle.image_bev is image_bev
    assert bundle.lidar_bev is lidar_bev
    assert bundle.fused_bev.shape == lidar_bev.shape
    assert bundle.head_feat[0] is bundle.fused_bev
    assert bundle.depth_aux is None


def test_lidar_proposal_generator_returns_common_pack():
    generator = LiDARProposalGenerator(
        in_channels=8,
        hidden_channel=16,
        num_classes=10,
        num_proposals=20,
    )
    proposals = generator(torch.randn(2, 8, 8, 8))
    assert isinstance(proposals, ProposalPack)
    assert proposals.query_feat_pre.shape == (2, 20, 16)
    assert proposals.query_feat_post.shape == (2, 20, 16)
    assert proposals.ref_xy.shape == (2, 20, 2)
    assert proposals.scores.shape == (2, 20)
    assert proposals.class_scores.shape == (2, 20, 10)
    assert proposals.labels.shape == (2, 20)
    assert proposals.indices.shape == (2, 20)
    assert proposals.dense_heatmap.shape == (2, 10, 8, 8)
    assert proposals.source_type == 'lidar'
    assert torch.isfinite(proposals.query_feat_post).all()
    assert torch.isfinite(proposals.ref_xy).all()
    assert proposals.ref_xy.min() >= 0.5
    assert proposals.ref_xy[..., 0].max() <= 7.5
    assert proposals.ref_xy[..., 1].max() <= 7.5

    proposals.query_feat_post.square().mean().backward()
    gradients = [
        parameter.grad for parameter in generator.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert any(
        torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        for gradient in gradients)


def test_proposal_utility_matches_legacy_selection():
    torch.manual_seed(7)
    feature = torch.randn(2, 16, 8, 8)
    dense_heatmap = torch.randn(2, 10, 8, 8)
    class_encoding = torch.nn.Conv1d(10, 16, 1)
    bev_pos = create_2d_grid(8, 8)
    pack = build_proposal_pack(
        feature,
        dense_heatmap,
        bev_pos,
        class_encoding,
        num_proposals=20,
        nms_kernel_size=3,
        dataset='nuScenes',
        source='fusion',
    )

    heatmap = dense_heatmap.detach().sigmoid()
    local_max = torch.zeros_like(heatmap)
    local_max[:, :, 1:-1, 1:-1] = torch.nn.functional.max_pool2d(
        heatmap, kernel_size=3, stride=1, padding=0)
    local_max[:, 8] = torch.nn.functional.max_pool2d(
        heatmap[:, 8], kernel_size=1, stride=1, padding=0)
    local_max[:, 9] = torch.nn.functional.max_pool2d(
        heatmap[:, 9], kernel_size=1, stride=1, padding=0)
    heatmap = (heatmap * (heatmap == local_max)).view(2, 10, -1)
    top = heatmap.view(2, -1).argsort(dim=-1, descending=True)[..., :20]
    labels = top // heatmap.shape[-1]
    indices = top % heatmap.shape[-1]
    expected_feat = feature.view(2, 16, -1).gather(
        2, indices[:, None].expand(-1, 16, -1))
    one_hot = torch.nn.functional.one_hot(labels, 10).permute(0, 2, 1)
    expected_feat += class_encoding(one_hot.float())
    expected_pos = bev_pos.repeat(2, 1, 1).gather(
        1, indices[:, :, None].expand(-1, -1, 2))

    assert torch.equal(pack.labels, labels)
    assert torch.equal(pack.indices, indices)
    expected_pre = feature.view(2, 16,
                                -1).gather(2, indices[:,
                                                      None].expand(-1, 16, -1))
    assert torch.equal(pack.query_feat_pre, expected_pre.transpose(1, 2))
    assert torch.equal(pack.query_feat_post, expected_feat.transpose(1, 2))
    assert torch.equal(pack.ref_xy, expected_pos)
    assert torch.equal(pack.scores, heatmap.view(2, -1).gather(1, top))


def test_transfusion_proposal_utility_is_side_effect_free():
    head = TransFusionHead.__new__(TransFusionHead)
    torch.nn.Module.__init__(head)
    head.shared_conv = torch.nn.Conv2d(16, 8, 1)
    head.heatmap_head = torch.nn.Conv2d(8, 10, 1)
    head.class_encoding = torch.nn.Conv1d(10, 8, 1)
    head.bev_pos = create_2d_grid(8, 8)
    head.num_proposals = 200
    head.nms_kernel_size = 3
    head.test_cfg = dict(dataset='nuScenes')
    head.eval()

    state_before = {
        key: value.detach().clone()
        for key, value in head.state_dict().items()
    }
    buffers_before = tuple(
        (name, id(buffer)) for name, buffer in head.named_buffers())
    first_input = torch.randn(2, 16, 8, 8)
    second_input = torch.randn(2, 16, 8, 8)
    pack_200 = head.extract_proposals(first_input, num_proposals=200)
    pack_300 = head.extract_proposals(second_input, num_proposals=300)
    pack_300_repeat = head.extract_proposals(second_input, num_proposals=300)

    assert pack_200.query_feat_post.shape == (2, 200, 8)
    assert pack_300.query_feat_pre.shape == (2, 300, 8)
    assert pack_300.query_feat_post.shape == (2, 300, 8)
    assert pack_300.ref_xy.shape == (2, 300, 2)
    assert pack_300.scores.shape == (2, 300)
    assert pack_300.labels.shape == (2, 300)
    assert all(
        torch.isfinite(tensor).all()
        for tensor in (pack_300.query_feat_pre, pack_300.query_feat_post,
                       pack_300.ref_xy, pack_300.scores))
    assert torch.equal(pack_300.query_feat_post,
                       pack_300_repeat.query_feat_post)
    assert torch.equal(pack_300.indices, pack_300_repeat.indices)
    assert not hasattr(head, 'query_labels')
    assert not hasattr(head, 'query_heatmap_score')
    assert tuple((name, id(buffer))
                 for name, buffer in head.named_buffers()) == buffers_before
    state_after = head.state_dict()
    assert state_before.keys() == state_after.keys()
    assert all(
        torch.equal(state_before[key], state_after[key])
        for key in state_before)


def test_module_and_parameter_ids_are_not_registered_twice():
    model = BEVFusion.__new__(BEVFusion)
    torch.nn.Module.__init__(model)
    model.fusion_layer = torch.nn.Conv2d(2, 2, 1)
    model.bbox_head = torch.nn.Conv2d(2, 2, 1)
    model.instance_refiner = LiDARProposalGenerator(
        in_channels=2,
        hidden_channel=4,
        num_classes=3,
        num_proposals=2,
        dataset='generic',
    )
    module_ids = [
        id(module) for _, module in model.named_modules(remove_duplicate=False)
    ]
    parameter_ids = [
        id(parameter)
        for _, parameter in model.named_parameters(remove_duplicate=False)
    ]
    assert len(module_ids) == len(set(module_ids))
    assert len(parameter_ids) == len(set(parameter_ids))
    keys = tuple(model.state_dict())
    assert not any(
        key.startswith('instance_refiner.fusion_layer.') for key in keys)
    assert not any(
        key.startswith('instance_refiner.bbox_head.') for key in keys)
    assert not any(
        key.startswith('instance_refiner.depth_supervisor.') for key in keys)
