# Copyright (c) OpenMMLab. All rights reserved.
# [module-C/DGF] Baseline-untouched / all-off consistency check.
#
# Module C is a pure config-level swap (ConvFuser -> DGFFuser); no runtime
# branch is added to the baseline. This script proves that switching to +C
# changes ONLY the fusion layer and leaves every other parameter tensor of the
# model identical (same state_dict keys), i.e. "+C off == baseline" holds by
# construction.
#
# Run from the repo ROOT on a machine where the BEVFusion CUDA ops are
# compiled (importing the project package loads ops/*.so):
#   python tools/check_c_alloff.py
#
# It builds the models on CPU (no GPU needed) and only compares state_dict
# KEYS, so pretrained weights are irrelevant -- we disable img_backbone
# `init_cfg` to avoid any (offline-unfriendly) checkpoint fetch.
import argparse

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.utils import import_modules_from_strings

from mmdet3d.registry import MODELS

BASE_CFG = ('projects/BEVFusion/configs/'
            'bevfusion_lidar-cam_voxel0075_4xa30-amp-accum_nus-3d.py')
DGF_CFG = ('projects/BEVFusion/configs/'
           'bevfusion_lidar-cam_4xa30_dgf_nus-3d.py')


def build_model(cfg_path):
    cfg = Config.fromfile(cfg_path)
    if cfg.get('custom_imports', None):
        import_modules_from_strings(**cfg['custom_imports'])
    init_default_scope('mmdet3d')
    # Compare keys only -> avoid fetching the Swin checkpoint (offline-safe).
    if cfg.model.get('img_backbone', None) is not None:
        cfg.model.img_backbone.init_cfg = None
    model = MODELS.build(cfg.model)
    return cfg, model


def fusion_keys(model):
    return {k for k in model.state_dict() if k.startswith('fusion_layer.')}


def non_fusion_keys(model):
    return {k for k in model.state_dict() if not k.startswith('fusion_layer.')}


def main():
    parser = argparse.ArgumentParser(
        description='Module-C baseline-untouched consistency check')
    parser.add_argument('--base', default=BASE_CFG)
    parser.add_argument('--dgf', default=DGF_CFG)
    args = parser.parse_args()

    base_cfg, base_model = build_model(args.base)
    dgf_cfg, dgf_model = build_model(args.dgf)

    # 1) config-level: baseline uses ConvFuser, +C uses DGFFuser
    assert base_cfg.model.fusion_layer.type == 'ConvFuser', \
        f'baseline fusion_layer is {base_cfg.model.fusion_layer.type}'
    assert dgf_cfg.model.fusion_layer.type == 'DGFFuser', \
        f'+C fusion_layer is {dgf_cfg.model.fusion_layer.type}'

    # 2) module-level: instantiated classes match
    assert type(base_model.fusion_layer).__name__ == 'ConvFuser'
    assert type(dgf_model.fusion_layer).__name__ == 'DGFFuser'

    # 3) everything OUTSIDE the fusion layer is byte-for-byte the same set of
    #    parameters -> module C touches only the fusion layer.
    base_nf, dgf_nf = non_fusion_keys(base_model), non_fusion_keys(dgf_model)
    only_base = base_nf - dgf_nf
    only_dgf = dgf_nf - base_nf
    assert base_nf == dgf_nf, (
        'Non-fusion parameter keys differ -> baseline was modified!\n'
        f'  only in baseline: {sorted(only_base)}\n'
        f'  only in +C:       {sorted(only_dgf)}')

    print('[check_c_alloff] PASS')
    print(f'  baseline fusion keys (ConvFuser): {len(fusion_keys(base_model))}')
    print(f'  +C       fusion keys (DGFFuser) : {len(fusion_keys(dgf_model))}')
    print(f'  non-fusion keys identical       : {len(base_nf)}')
    print('  => module C changes ONLY model.fusion_layer; baseline untouched.')


if __name__ == '__main__':
    main()
