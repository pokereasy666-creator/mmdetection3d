# -*- coding: utf-8 -*-
"""QueryLocBEVFusion: BEVFusion with the exec doc #3 freezing protocol.

Frozen (requires_grad False AND eval mode, CLAUDE.md rules 1 / 10): camera
branch (img_backbone, img_neck, view_transform) and LiDAR branch
(pts_voxel_layer, pts_voxel_encoder, pts_middle_encoder). mmengine calls
model.train() at every epoch start and after validation, so eval mode is
re-applied in train(); modules are looked up by name at call time, which
also covers SyncBN conversion (it replaces BN modules after construction).
"""
from mmdet3d.registry import MODELS
from projects.BEVFusion.bevfusion.bevfusion import BEVFusion

DEFAULT_FROZEN = ('img_backbone', 'img_neck', 'view_transform',
                  'pts_voxel_layer', 'pts_voxel_encoder',
                  'pts_middle_encoder')


def freeze_modules(model, names):
    """requires_grad_(False) on every named submodule; returns the count of
    frozen parameters. Missing names raise (a typo must not unfreeze)."""
    n = 0
    for name in names:
        m = getattr(model, name)
        if m is None:
            raise ValueError('frozen module %r is None' % name)
        for p in m.parameters():
            p.requires_grad_(False)
            n += p.numel()
    return n


def set_frozen_eval(model, names):
    for name in names:
        getattr(model, name).eval()


@MODELS.register_module()
class QueryLocBEVFusion(BEVFusion):

    def __init__(self, *args, frozen_modules=DEFAULT_FROZEN, **kwargs):
        super().__init__(*args, **kwargs)
        self.frozen_modules = tuple(frozen_modules)
        self.num_frozen_params = freeze_modules(self, self.frozen_modules)
        set_frozen_eval(self, self.frozen_modules)

    def train(self, mode=True):
        super().train(mode)
        set_frozen_eval(self, self.frozen_modules)
        return self
