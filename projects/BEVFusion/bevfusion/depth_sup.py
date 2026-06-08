# Copyright (c) OpenMMLab. All rights reserved.
# [module-A/depth-sup] BEVDepth (arXiv:2206.10092) explicit depth-supervision
# helpers.
#
# Self-contained (torch only -- NO project ops, NO mmcv/mmdet), so they can be
# unit-tested on CPU without the compiled BEVFusion CUDA extensions. Used by
# `DepthLSSTransform.get_depth_loss` (depth_lss.py) when `use_depth_sup=True`.
# These functions add NO learnable parameters: they only supervise the depth
# logits the baseline `depthnet` already predicts.
from typing import Tuple

import torch
import torch.nn.functional as F


def downsample_gt_depth(
    gt_depth: torch.Tensor,
    image_size: Tuple[int, int],
    feature_size: Tuple[int, int],
    dbound: Tuple[float, float, float],
    num_bins: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse LiDAR depth GT -> per-feature-cell one-hot depth bins + valid mask.

    Re-implements BEVDepth's `get_downsampled_gt_depth` (min-pool the nearest
    non-zero depth in each (ds_h x ds_w) patch, then discretize per `dbound`),
    self-contained.

    Args:
        gt_depth (Tensor): ``(M, 1, iH, iW)`` sparse depth (0 where no point).
        image_size (tuple): ``(iH, iW)`` of ``gt_depth``.
        feature_size (tuple): ``(fH, fW)`` of the predicted depth logits.
        dbound (tuple): ``[d_min, d_max, d_step]`` (e.g. ``[1.0, 60.0, 0.5]``).
        num_bins (int): number of depth bins ``D``.

    Returns:
        one_hot (Tensor): ``(M, fH, fW, D)`` float one-hot bin labels.
        valid (Tensor): ``(M, fH, fW)`` bool. True ONLY where a LiDAR point
            lands inside ``[d_min, d_max)``; no-point pixels are False so they
            are never supervised (the sparse mask).
    """
    M = gt_depth.shape[0]
    iH, iW = image_size
    fH, fW = feature_size
    assert iH % fH == 0 and iW % fW == 0, \
        f'image_size {image_size} not divisible by feature_size {feature_size}'
    ds_h, ds_w = iH // fH, iW // fW
    # group each (ds_h, ds_w) image patch into the last dim
    g = gt_depth.view(M, fH, ds_h, fW, ds_w, 1)
    g = g.permute(0, 1, 3, 5, 2, 4).contiguous().view(-1, ds_h * ds_w)
    # keep the nearest NON-zero depth in each patch (0 -> +inf loses the min)
    g = torch.where(g == 0.0, torch.full_like(g, 1e5), g)
    g = torch.min(g, dim=-1).values.view(M, fH, fW)
    d_min, d_max, d_step = dbound
    # a pixel is supervised only if a LiDAR point fell in the valid depth range
    valid = (g >= d_min) & (g < d_max)
    bin_idx = ((g - d_min) / d_step).long().clamp(0, num_bins - 1)
    one_hot = F.one_hot(bin_idx, num_classes=num_bins).float()
    return one_hot, valid


def depth_bce_loss(
    logits: torch.Tensor,
    gt_depth: torch.Tensor,
    image_size: Tuple[int, int],
    feature_size: Tuple[int, int],
    dbound: Tuple[float, float, float],
    num_bins: int,
    weight: float = 0.5,
) -> torch.Tensor:
    """Masked per-bin BCE-with-logits depth loss (BEVDepth-style).

    Args:
        logits (Tensor): ``(M, D, fH, fW)`` PRE-softmax depth logits (BCE needs
            logits, not the softmaxed probabilities).
        gt_depth (Tensor): ``(M, 1, iH, iW)`` sparse LiDAR depth GT.
        image_size / feature_size / dbound / num_bins: see ``downsample_gt_depth``.
        weight (float): loss weight (``depth_loss_weight``).

    Returns:
        Tensor: scalar ``weight * mean_BCE`` over valid pixels, or a graph-
        connected zero if there is no valid LiDAR pixel in the batch.
    """
    one_hot, valid = downsample_gt_depth(gt_depth, image_size, feature_size,
                                         dbound, num_bins)
    # (M, D, fH, fW) -> (M, fH, fW, D), then keep only valid pixels (the mask)
    pred = logits.permute(0, 2, 3, 1)[valid]   # (Nvalid, D)
    tgt = one_hot[valid]                        # (Nvalid, D)
    if pred.numel() == 0:
        # no LiDAR-hit pixel this batch -> zero loss, but keep the graph alive
        return logits.sum() * 0.0
    # per-bin sigmoid BCE on logits (autocast-safe, numerically stable).
    loss = F.binary_cross_entropy_with_logits(pred, tgt, reduction='mean')
    return weight * loss
