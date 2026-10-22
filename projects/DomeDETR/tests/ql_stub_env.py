# -*- coding: utf-8 -*-
"""CPU stand-ins for mmcv / mmdet / mmdet3d (dev-side tests only).

mmcv's compiled ops cannot be installed here, so the library pieces the
TransFusion head touches are replaced by small modules whose semantics
follow mmcv 2.1 / mmdet 3.2 (ConvModule, SeparateHead, AssignResult,
PseudoSampler, multi_apply, FocalLoss / GaussianFocalLoss / L1Loss with
avg_factor, FocalLossCost, a BEV-IoU calculator, a DETR decoder layer).
The files under test are loaded UNCHANGED from the repo:
  projects/BEVFusion/bevfusion/transfusion_head.py, transformer.py, utils.py
  (real TransFusionHead, TransformerDecoderLayer, TransFusionBBoxCoder,
  HungarianAssigner3D, BBoxBEVL1Cost, IoU3DCost),
  mmdet3d/models/utils/gaussian.py (draw_heatmap_gaussian, gaussian_radius),
  projects/QueryLoc/queryloc/{subcell,head,model}.py.
Because the original head and the QueryLoc subclass run on the same
stand-ins, comparisons between them test the subclass plumbing; the real
libraries are exercised by the server checks (S-T0, S-M0, S-M1).
"""
import importlib.util
import os.path as osp
import sys
import types
from functools import partial

import torch
import torch.nn.functional as F
from mmengine.config import ConfigDict
from mmengine.registry import Registry
from torch import nn

REPO = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))
_ENV = {}


def _mod(name, pkg=False, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    if pkg:
        m.__path__ = []
    sys.modules[name] = m
    return m


def _load(name, path, package=None):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    if package is not None:
        m.__package__ = package
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# --------------------------------------------------------------------- mmcv
class ConvModule(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias='auto', conv_cfg=None, norm_cfg=None,
                 act_cfg=dict(type='ReLU'), **kwargs):
        super().__init__()
        ctype = (conv_cfg or {}).get('type', 'Conv2d')
        if bias == 'auto':
            bias = norm_cfg is None
        conv = nn.Conv1d if ctype == 'Conv1d' else nn.Conv2d
        self.conv = conv(in_channels, out_channels, kernel_size, stride,
                         padding, bias=bool(bias))
        self.bn = None
        if norm_cfg is not None:
            bn = nn.BatchNorm1d if norm_cfg['type'] in ('BN1d', ) else \
                nn.BatchNorm2d
            self.bn = bn(out_channels)
        self.act = nn.ReLU(inplace=True) if act_cfg is not None else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return self.act(x) if self.act is not None else x


def build_conv_layer(cfg, *args, **kwargs):
    ctype = (cfg or {}).get('type', 'Conv2d')
    if isinstance(kwargs.get('bias'), str):
        kwargs['bias'] = True
    return (nn.Conv1d if ctype == 'Conv1d' else nn.Conv2d)(*args, **kwargs)


# --------------------------------------------------------------------- mmdet
def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    return tuple(map(list, zip(*map(pfunc, *args))))


class AssignResult:

    def __init__(self, num_gts, gt_inds, max_overlaps, labels=None):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels


class BaseAssigner:
    pass


class BaseBBoxCoder:

    def __init__(self, **kwargs):
        pass


class _SamplingResult:
    pass


class PseudoSampler:

    def __init__(self, **kwargs):
        pass

    def sample(self, assign_result, pred_instances, gt_instances, *a, **k):
        gt = gt_instances.bboxes
        pos = torch.nonzero(assign_result.gt_inds > 0,
                            as_tuple=False).squeeze(-1).unique()
        neg = torch.nonzero(assign_result.gt_inds == 0,
                            as_tuple=False).squeeze(-1).unique()
        r = _SamplingResult()
        r.pos_inds, r.neg_inds = pos, neg
        r.pos_assigned_gt_inds = assign_result.gt_inds[pos] - 1
        r.pos_gt_bboxes = gt[r.pos_assigned_gt_inds.long()] if gt.numel() \
            else gt.view(-1, gt.shape[-1])
        return r


def _reduce(loss, weight=None, reduction='mean', avg_factor=None):
    if weight is not None:
        loss = loss * weight
    if avg_factor is None:
        return loss.mean() if reduction == 'mean' else loss.sum()
    return loss.sum() / (avg_factor + torch.finfo(torch.float32).eps)


class GaussianFocalLoss(nn.Module):

    def __init__(self, alpha=2.0, gamma=4.0, reduction='mean',
                 loss_weight=1.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
        self.reduction, self.loss_weight = reduction, loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None):
        eps = 1e-12
        pos = target.eq(1)
        neg_w = (1 - target).pow(self.gamma)
        loss = -(pred + eps).log() * (1 - pred).pow(self.alpha) * pos - \
            (1 - pred + eps).log() * pred.pow(self.alpha) * neg_w
        return self.loss_weight * _reduce(loss, weight, self.reduction,
                                          avg_factor)


class FocalLoss(nn.Module):

    def __init__(self, use_sigmoid=True, gamma=2.0, alpha=0.25,
                 reduction='mean', loss_weight=1.0):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha
        self.reduction, self.loss_weight = reduction, loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None):
        C = pred.shape[1]
        t = F.one_hot(target, num_classes=C + 1)[:, :C].type_as(pred)
        p = pred.sigmoid()
        pt = (1 - p) * t + p * (1 - t)
        fw = (self.alpha * t + (1 - self.alpha) * (1 - t)) * pt.pow(self.gamma)
        loss = F.binary_cross_entropy_with_logits(pred, t,
                                                  reduction='none') * fw
        if weight is not None:
            weight = weight.view(-1, 1).float()
        return self.loss_weight * _reduce(loss, weight, self.reduction,
                                          avg_factor)


class L1Loss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction, self.loss_weight = reduction, loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None):
        return self.loss_weight * _reduce((pred - target).abs(), weight,
                                          self.reduction, avg_factor)


class FocalLossCost:

    def __init__(self, weight=1.0, alpha=0.25, gamma=2, eps=1e-12):
        self.weight, self.alpha, self.gamma, self.eps = \
            weight, alpha, gamma, eps

    def __call__(self, pred_instances, gt_instances, *a, **k):
        p = pred_instances.scores.sigmoid()
        neg = -(1 - p + self.eps).log() * (1 - self.alpha) * \
            p.pow(self.gamma)
        pos = -(p + self.eps).log() * self.alpha * (1 - p).pow(self.gamma)
        lab = gt_instances.labels
        return (pos[:, lab] - neg[:, lab]) * self.weight


class BboxOverlaps3D:
    """Axis-aligned BEV IoU (stand-in for the rotated 3D IoU)."""

    def __init__(self, coordinate='lidar'):
        pass

    def __call__(self, a, b):
        def corners(x):
            return (x[:, 0] - x[:, 3] / 2, x[:, 1] - x[:, 4] / 2,
                    x[:, 0] + x[:, 3] / 2, x[:, 1] + x[:, 4] / 2)
        ax0, ay0, ax1, ay1 = corners(a)
        bx0, by0, bx1, by1 = corners(b)
        iw = (torch.min(ax1[:, None], bx1[None]) -
              torch.max(ax0[:, None], bx0[None])).clamp(min=0)
        ih = (torch.min(ay1[:, None], by1[None]) -
              torch.max(ay0[:, None], by0[None])).clamp(min=0)
        inter = iw * ih
        area_a = ((ax1 - ax0) * (ay1 - ay0))[:, None]
        area_b = ((bx1 - bx0) * (by1 - by0))[None]
        return inter / (area_a + area_b - inter).clamp(min=1e-6)


class _MHA(nn.Module):

    def __init__(self, embed_dims, num_heads, dropout=0.0, **kwargs):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dims, num_heads,
                                          dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, query, key=None, value=None, query_pos=None,
                key_pos=None, attn_mask=None, key_padding_mask=None,
                **kwargs):
        q = query if query_pos is None else query + query_pos
        k = key if key_pos is None else key + key_pos
        out = self.attn(q, k, value, attn_mask=attn_mask,
                        key_padding_mask=key_padding_mask)[0]
        return query + self.drop(out)


class _FFN(nn.Module):

    def __init__(self, embed_dims, feedforward_channels, ffn_drop=0.0,
                 **kwargs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dims, feedforward_channels), nn.ReLU(),
            nn.Dropout(ffn_drop), nn.Linear(feedforward_channels, embed_dims),
            nn.Dropout(ffn_drop))

    def forward(self, x):
        return x + self.layers(x)


class DetrTransformerDecoderLayer(nn.Module):

    def __init__(self, self_attn_cfg, cross_attn_cfg, ffn_cfg,
                 norm_cfg=dict(type='LN'), init_cfg=None):
        super().__init__()
        self.self_attn = _MHA(**self_attn_cfg)
        self.cross_attn = _MHA(**cross_attn_cfg)
        self.ffn = _FFN(**ffn_cfg)
        d = self_attn_cfg['embed_dims']
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])


class SeparateHead(nn.Module):

    def __init__(self, in_channels, heads, head_conv=64, final_kernel=1,
                 init_bias=-2.19, conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d'), bias='auto', init_cfg=None,
                 **kwargs):
        super().__init__()
        self.heads = heads
        for head, (classes, num_conv) in heads.items():
            layers, c_in = [], in_channels
            for _ in range(num_conv - 1):
                layers.append(ConvModule(c_in, head_conv, final_kernel,
                                         padding=final_kernel // 2,
                                         bias=bias, conv_cfg=conv_cfg,
                                         norm_cfg=norm_cfg))
                c_in = head_conv
            layers.append(build_conv_layer(conv_cfg, head_conv, classes,
                                           kernel_size=final_kernel,
                                           padding=final_kernel // 2,
                                           bias=True))
            self.__setattr__(head, nn.Sequential(*layers))
            if head == 'heatmap':
                layers[-1].bias.data.fill_(init_bias)

    def forward(self, x):
        return {h: self.__getattr__(h)(x) for h in self.heads}


class Boxes:
    """Minimal LiDARInstance3DBoxes: [N, 9] = x y z(bottom) dx dy dz yaw
    vx vy."""

    def __init__(self, t):
        self.tensor = t

    @property
    def gravity_center(self):
        c = self.tensor[:, :3].clone()
        c[:, 2] = c[:, 2] + self.tensor[:, 5] * 0.5
        return c

    def __len__(self):
        return self.tensor.shape[0]


class BEVFusionStub(nn.Module):
    """Stand-in BEVFusion with the module names QueryLocBEVFusion freezes."""

    def __init__(self, **kwargs):
        super().__init__()

        def block():
            return nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4))
        for n in ('img_backbone', 'img_neck', 'view_transform',
                  'pts_voxel_encoder', 'pts_middle_encoder', 'fusion_layer',
                  'pts_backbone', 'pts_neck', 'bbox_head'):
            setattr(self, n, block())
        self.pts_voxel_layer = nn.Identity()


def install():
    """Install the stand-ins and load the files under test (idempotent)."""
    if _ENV:
        return _ENV
    MODELS, TASK_UTILS = Registry('model'), Registry('task util')
    for n, c in (('GaussianFocalLoss', GaussianFocalLoss),
                 ('FocalLoss', FocalLoss), ('L1Loss', L1Loss)):
        MODELS.register_module(n, module=c)
    TASK_UTILS.register_module('FocalLossCost', module=FocalLossCost)
    TASK_UTILS.register_module('BboxOverlaps3D', module=BboxOverlaps3D)
    TASK_UTILS.register_module('PseudoSampler', module=PseudoSampler)
    _mod('mmcv', pkg=True)
    _mod('mmcv.cnn', ConvModule=ConvModule, build_conv_layer=build_conv_layer)
    _mod('mmdet', pkg=True)
    _mod('mmdet.models', pkg=True,
         DetrTransformerDecoderLayer=DetrTransformerDecoderLayer)
    _mod('mmdet.models.task_modules', AssignResult=AssignResult,
         PseudoSampler=PseudoSampler, BaseAssigner=BaseAssigner,
         BaseBBoxCoder=BaseBBoxCoder, build_assigner=TASK_UTILS.build,
         build_bbox_coder=TASK_UTILS.build, build_sampler=TASK_UTILS.build)
    _mod('mmdet.models.utils', multi_apply=multi_apply)
    _mod('mmdet3d', pkg=True)
    _mod('mmdet3d.registry', MODELS=MODELS, TASK_UTILS=TASK_UTILS)
    gauss = _load('mmdet3d.models.utils.gaussian',
                  osp.join(REPO, 'mmdet3d/models/utils/gaussian.py'))
    _mod('mmdet3d.models', pkg=True, circle_nms=None,
         draw_heatmap_gaussian=gauss.draw_heatmap_gaussian,
         gaussian_radius=gauss.gaussian_radius)
    _mod('mmdet3d.models.dense_heads', pkg=True)
    _mod('mmdet3d.models.dense_heads.centerpoint_head',
         SeparateHead=SeparateHead)
    _mod('mmdet3d.models.layers', nms_bev=None)
    _mod('mmdet3d.structures', xywhr2xyxyr=None)
    _mod('projects', pkg=True)
    _mod('projects.BEVFusion', pkg=True)
    _mod('projects.BEVFusion.bevfusion', pkg=True)
    bev = osp.join(REPO, 'projects/BEVFusion/bevfusion')
    th = _load('projects.BEVFusion.bevfusion.transfusion_head',
               osp.join(bev, 'transfusion_head.py'))
    _load('projects.BEVFusion.bevfusion.transformer',
          osp.join(bev, 'transformer.py'))
    _load('projects.BEVFusion.bevfusion.utils', osp.join(bev, 'utils.py'))
    _mod('projects.BEVFusion.bevfusion.bevfusion', BEVFusion=BEVFusionStub)
    _mod('projects.QueryLoc', pkg=True)
    _mod('projects.QueryLoc.queryloc', pkg=True)
    ql = osp.join(REPO, 'projects/QueryLoc/queryloc')
    pkg = 'projects.QueryLoc.queryloc'
    sc = _load(pkg + '.subcell', osp.join(ql, 'subcell.py'), pkg)
    hd = _load(pkg + '.head', osp.join(ql, 'head.py'), pkg)
    md = _load(pkg + '.model', osp.join(ql, 'model.py'), pkg)
    _ENV.update(MODELS=MODELS, TASK_UTILS=TASK_UTILS,
                TransFusionHead=th.TransFusionHead,
                QueryLocTransFusionHead=hd.QueryLocTransFusionHead,
                QueryLocBEVFusion=md.QueryLocBEVFusion, model_mod=md,
                subcell=sc, gauss=gauss, Boxes=Boxes)
    return _ENV


# --------------------------------------------------------------- test head
GRID = 24          # BEV cells per side (0.6 m each)
PC = GRID * 0.3    # half extent in metres


def head_cfg(**over):
    """A small head with the real config structure (nuScenes test_cfg so
    the class 8/9 no-NMS rule is exercised)."""
    cfg = dict(
        type='TransFusionHead',
        num_proposals=20,
        auxiliary=True,
        in_channels=16,
        hidden_channel=32,
        num_classes=10,
        nms_kernel_size=3,
        bn_momentum=0.1,
        num_decoder_layers=1,
        decoder_layer=dict(
            type='TransformerDecoderLayer',
            self_attn_cfg=dict(embed_dims=32, num_heads=4, dropout=0.1),
            cross_attn_cfg=dict(embed_dims=32, num_heads=4, dropout=0.1),
            ffn_cfg=dict(embed_dims=32, feedforward_channels=64, num_fcs=2,
                         ffn_drop=0.1),
            norm_cfg=dict(type='LN'),
            pos_encoding_cfg=dict(input_channel=2, num_pos_feats=32)),
        train_cfg=ConfigDict(
            dataset='nuScenes',
            point_cloud_range=[-PC, -PC, -5.0, PC, PC, 3.0],
            grid_size=[GRID * 8, GRID * 8, 41],
            voxel_size=[0.075, 0.075, 0.2],
            out_size_factor=8,
            gaussian_overlap=0.1,
            min_radius=2,
            pos_weight=-1,
            code_weights=[1.0] * 8 + [0.2, 0.2],
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(type='BboxOverlaps3D'),
                cls_cost=dict(type='FocalLossCost', gamma=2.0, alpha=0.25,
                              weight=0.15),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25))),
        test_cfg=dict(
            dataset='nuScenes',
            grid_size=[GRID * 8, GRID * 8, 41],
            out_size_factor=8,
            voxel_size=[0.075, 0.075],
            pc_range=[-PC, -PC],
            nms_type=None),
        common_heads=dict(
            center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2],
            vel=[2, 2]),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=[-PC, -PC],
            post_center_range=[-PC - 6, -PC - 6, -10.0, PC + 6, PC + 6, 10.0],
            score_threshold=0.0,
            out_size_factor=8,
            voxel_size=[0.075, 0.075],
            code_size=10),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0,
                      alpha=0.25, reduction='mean', loss_weight=1.0),
        loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean',
                          loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25))
    cfg.update(over)
    return cfg


def make_gt(n, seed, cell_centres=False):
    """n GT boxes inside the map; labels 0..9; returns InstanceData."""
    from mmengine.structures import InstanceData
    g = torch.Generator().manual_seed(seed)
    if cell_centres:
        idx = torch.randperm(GRID * GRID, generator=g)[:n]
        r, c = idx // GRID, idx % GRID
        x = (r.float() + 0.5) * 0.6 - PC
        y = (c.float() + 0.5) * 0.6 - PC
    else:
        x = (torch.rand(n, generator=g) * 2 - 1) * (PC - 0.5)
        y = (torch.rand(n, generator=g) * 2 - 1) * (PC - 0.5)
    t = torch.zeros(n, 9)
    t[:, 0], t[:, 1] = x, y
    t[:, 2] = -1.0
    t[:, 3] = 1.0 + 3.0 * torch.rand(n, generator=g)
    t[:, 4] = 0.6 + 1.5 * torch.rand(n, generator=g)
    t[:, 5] = 1.5
    t[:, 6] = (torch.rand(n, generator=g) * 2 - 1) * 3.0
    t[:, 7:] = torch.randn(n, 2, generator=g)
    lab = torch.randint(0, 10, (n, ), generator=g)
    return InstanceData(bboxes_3d=Boxes(t), labels_3d=lab)
