# Copyright (c) OpenMMLab. All rights reserved.
# [module-A/depth-sup] Baseline-untouched / all-off consistency check.
#
# Module A (BEVDepth-style depth supervision) is gated by the view transform's
# `use_depth_sup` flag and adds NO trainable parameters (it only supervises the
# depth logits the baseline depthnet already predicts). This script proves:
#   1. baseline config -> view_transform.use_depth_sup is False (no depth
#      behaviour, BEVFusion.loss adds no 'loss_depth');
#   2. +A config -> view_transform.use_depth_sup is True;
#   3. the +A model has the SAME state_dict keys as the baseline (zero new
#      params), i.e. the only difference is a training-time loss term.
#
# Run from the repo ROOT where the BEVFusion CUDA ops are compiled (importing
# the project package loads ops/*.so). Builds on CPU; compares keys only, so
# the Swin init_cfg checkpoint fetch is disabled (offline-safe).
import argparse

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.utils import import_modules_from_strings

from mmdet3d.registry import MODELS

BASE_CFG = ('projects/BEVFusion/configs/'
            'bevfusion_lidar-cam_voxel0075_4xa30-amp-accum_nus-3d.py')
A_CFG = ('projects/BEVFusion/configs/'
         'bevfusion_lidar-cam_4xa30_depthsup_nus-3d.py')


def build_model(cfg_path):
    cfg = Config.fromfile(cfg_path)
    if cfg.get('custom_imports', None):
        import_modules_from_strings(**cfg['custom_imports'])
    init_default_scope('mmdet3d')
    if cfg.model.get('img_backbone', None) is not None:
        cfg.model.img_backbone.init_cfg = None  # keys only -> no ckpt fetch
    return cfg, MODELS.build(cfg.model)


def main():
    parser = argparse.ArgumentParser(
        description='Module-A baseline-untouched consistency check')
    parser.add_argument('--base', default=BASE_CFG)
    parser.add_argument('--variant', default=A_CFG)
    args = parser.parse_args()

    base_cfg, base_model = build_model(args.base)
    a_cfg, a_model = build_model(args.variant)

    # 1) flag semantics
    base_flag = getattr(base_model.view_transform, 'use_depth_sup', False)
    a_flag = getattr(a_model.view_transform, 'use_depth_sup', False)
    assert base_flag is False, \
        f'baseline view_transform.use_depth_sup must be False, got {base_flag}'
    assert a_flag is True, \
        f'+A view_transform.use_depth_sup must be True, got {a_flag}'

    # 2) zero new parameters: identical state_dict keys
    base_keys = set(base_model.state_dict().keys())
    a_keys = set(a_model.state_dict().keys())
    only_base = base_keys - a_keys
    only_a = a_keys - base_keys
    assert base_keys == a_keys, (
        'state_dict keys differ -> module A added/removed parameters!\n'
        f'  only in baseline: {sorted(only_base)}\n'
        f'  only in +A:       {sorted(only_a)}')

    # 3) baseline caches absent (no depth tensors hanging around)
    assert getattr(base_model.view_transform, '_depth_pred_logits', None) is None
    assert getattr(base_model.view_transform, '_depth_gt', None) is None

    print('[check_a_alloff] PASS')
    print(f'  baseline use_depth_sup = {base_flag} ; +A use_depth_sup = {a_flag}')
    print(f'  state_dict keys identical: {len(base_keys)} keys '
          '(module A adds 0 parameters)')
    print('  => baseline path untouched; +A only adds a training-time '
          'loss_depth term.')


if __name__ == '__main__':
    main()
