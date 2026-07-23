# Copyright (c) OpenMMLab. All rights reserved.
# [module-A/depth-sup] Occlusion probe: does the depth-supervised model INFER
# depth, or does it COPY the input LiDAR depth (the leakage shortcut)?
#
# Why: the stock DepthLSSTransform feeds the SAME sparse LiDAR depth as both the
# depthnet INPUT and (under v1 supervision) the GT TARGET, and only observed
# pixels are supervised. A model can then minimise the depth loss by copying the
# input at supervised pixels instead of learning to infer depth. This tool
# MEASURES that: it hides a fixed fraction of the input LiDAR points, then, on
# the feature cells that DO have a full-projection GT, compares predicted-depth
# accuracy between:
#   - KEPT cells    : the point is still in the (occluded) input  -> copy works
#   - HELD-OUT cells: the point was removed from the input        -> only
#                     inference works (GT still known for scoring)
# A copy/shortcut model shows a LARGE accuracy gap (kept >> held-out). A model
# that genuinely infers depth shows a SMALL gap. Run it on the baseline and on
# a depth-supervised checkpoint and compare the gaps.
#
# This is a read-only analysis tool (NO training, NO grad). It monkeypatches the
# view transform's `get_cam_feats` with a probe version that applies the fixed
# occlusion and records (full GT, kept-input mask, predicted logits); the ~10
# lifted lines are kept in sync with DepthLSSTransform.get_cam_feats and marked.
#
# NOTE: this script needs a GPU, the nuScenes val data, and the compiled
# BEVFusion ops -- it has NOT been executed in the review sandbox. Run it on the
# 4xA30 server, e.g.:
#   python projects/BEVFusion/tools/probe_depth_shortcut.py \
#       --config projects/BEVFusion/configs/\
# bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
#       --checkpoint <BASELINE_or_+A_ckpt>.pth \
#       --num-batches 50 --occlude 0.5 --out outputs/probe_<name>.json
import argparse
import json
import os
import types

import torch

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint

# Bare-script import shim (repo root is 3 levels up), matching viz_depth_gt.py.
import sys  # noqa: E402
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import projects.BEVFusion.bevfusion  # noqa: F401,E402  (register modules)
from projects.BEVFusion.bevfusion.depth_sup import (  # noqa: E402
    downsample_gt_depth)
from mmdet3d.registry import MODELS  # noqa: E402


def _make_probe_get_cam_feats(occlude: float):
    """Return a `get_cam_feats` that occludes the input and records tensors.

    Uses the module's own trained `dtransform`/`depthnet`, so the prediction is
    faithful; only the input occlusion + recording is added. The lifted block is
    a copy of DepthLSSTransform.get_cam_feats (keep in sync).
    """
    keep_ratio = 1.0 - occlude

    def probe_get_cam_feats(self, x, d):
        B, N, C, fH, fW = x.shape
        d = d.view(B * N, *d.shape[2:])
        x = x.view(B * N, C, fH, fW)

        # record the FULL projection (scoring GT) before occluding the input
        self._probe_gt_full = d.detach().clone()      # (B*N, 1, iH, iW)
        keep = torch.rand_like(d) < keep_ratio
        # which real input points survive the occlusion (for the kept/held split)
        self._probe_input_kept = ((d > 0) & keep).detach()
        d = d * keep.to(d.dtype)                       # occluded depthnet input

        # ---- lifted from DepthLSSTransform.get_cam_feats (keep in sync) -------
        d = self.dtransform(d)
        x = torch.cat([d, x], dim=1)
        x = self.depthnet(x)
        self._probe_logits = x[:, :self.D].detach()    # (B*N, D, fH, fW)
        depth = x[:, :self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)
        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        # -----------------------------------------------------------------------
        return x

    return probe_get_cam_feats


@torch.no_grad()
def _accumulate(vt, dbound, acc):
    """Score the last forward's stashed tensors into `acc` (running sums)."""
    image_size = vt.image_size
    feature_size = vt.feature_size
    D = vt.D
    gt_full = vt._probe_gt_full
    logits = vt._probe_logits
    input_kept = vt._probe_input_kept.to(gt_full.dtype)

    # GT bins + valid mask at feature resolution (same as training)
    one_hot, valid = downsample_gt_depth(gt_full, image_size, feature_size,
                                         dbound, D)
    gt_bin = one_hot.argmax(dim=-1)                     # (M, fH, fW)
    pred_bin = logits.argmax(dim=1)                     # (M, fH, fW)

    # a valid GT cell is "kept" if the OCCLUDED input still has a point there
    _, valid_kept = downsample_gt_depth(gt_full * input_kept, image_size,
                                        feature_size, dbound, D)
    kept = valid & valid_kept                           # answer still in input
    held = valid & (~valid_kept)                        # removed from input

    abs_err = (pred_bin - gt_bin).abs().float()
    for name, mask in (('kept', kept), ('held', held)):
        n = int(mask.sum())
        if n == 0:
            continue
        acc[name]['n'] += n
        acc[name]['hit1'] += int(((abs_err == 0) & mask).sum())
        acc[name]['hit2'] += int(((abs_err <= 2) & mask).sum())
        acc[name]['abs_bin_err'] += float(abs_err[mask].sum())


def main():
    p = argparse.ArgumentParser(
        description='Occlusion probe for the depth-supervision leakage shortcut')
    p.add_argument(
        '--config',
        default='projects/BEVFusion/configs/'
        'bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--num-batches', type=int, default=50)
    p.add_argument('--occlude', type=float, default=0.5,
                   help='fraction of input LiDAR points to hide (0..1)')
    p.add_argument('--out', default=None, help='optional JSON output path')
    args = p.parse_args()
    assert 0.0 < args.occlude < 1.0, '--occlude must be in (0, 1)'

    init_default_scope('mmdet3d')
    cfg = Config.fromfile(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval().to(device)
    vt = model.view_transform
    assert vt is not None, 'model has no view_transform (need the camera branch)'
    vt.get_cam_feats = types.MethodType(
        _make_probe_get_cam_feats(args.occlude), vt)

    loader = Runner.build_dataloader(cfg.test_dataloader)
    acc = {k: dict(n=0, hit1=0, hit2=0, abs_bin_err=0.0)
           for k in ('kept', 'held')}

    seen = 0
    for batch in loader:
        if seen >= args.num_batches:
            break
        batch = model.data_preprocessor(batch, False)
        # trigger the camera branch (and thus the probe hook); mirror predict():
        # extract_feat(batch_inputs_dict, batch_input_metas).
        metas = [ds.metainfo for ds in batch['data_samples']]
        model.extract_feat(batch['inputs'], metas)
        _accumulate(vt, tuple(vt.dbound), acc)
        seen += 1
    print(f'[probe] scored {seen} batches, occlude={args.occlude}')

    def _summ(g):
        n = max(1, g['n'])
        return dict(cells=g['n'], acc_exact=g['hit1'] / n,
                    acc_within2=g['hit2'] / n, mean_abs_bin_err=g['abs_bin_err'] / n)

    kept, held = _summ(acc['kept']), _summ(acc['held'])
    gap = dict(
        d_acc_exact=kept['acc_exact'] - held['acc_exact'],
        d_acc_within2=kept['acc_within2'] - held['acc_within2'],
        d_mean_abs_bin_err=held['mean_abs_bin_err'] - kept['mean_abs_bin_err'])
    report = dict(checkpoint=args.checkpoint, occlude=args.occlude,
                  batches=seen, kept=kept, held=held, gap=gap)

    print(json.dumps(report, indent=2))
    print('\n[probe] interpretation: a LARGE kept-minus-held accuracy gap '
          '(d_acc_*) => the model COPIES input depth (shortcut). A small gap '
          '=> it genuinely infers depth. Compare baseline vs +A.')
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(report, f, indent=2)
        print(f'[probe] wrote {args.out}')


if __name__ == '__main__':
    main()
