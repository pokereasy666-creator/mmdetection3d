# Copyright (c) OpenMMLab. All rights reserved.
# [module-A/depth-sup] CPU unit tests for BEVDepth-style depth supervision.
#
# The core tests exercise depth_sup.py (torch-only) and run on pure CPU without
# the compiled BEVFusion ops. We import depth_sup.py *standalone* (via sys.path)
# to avoid the package __init__ -> depth_lss -> ops/*.so import chain.
#
# The DepthLSSTransform-level tests need the project package (-> compiled ops);
# they are skipped automatically where the ops are unavailable.
#
# Run from the repo root:  pytest projects/BEVFusion/tests/test_depth_sup.py -q
import os
import sys

import pytest
import torch
import torch.nn.functional as F

_BEVFUSION_DIR = os.path.join(os.path.dirname(__file__), '..', 'bevfusion')
sys.path.insert(0, os.path.abspath(_BEVFUSION_DIR))
import depth_sup  # noqa: E402

DBOUND = [1.0, 60.0, 0.5]          # -> D = 118 bins
NUM_BINS = int((DBOUND[1] - DBOUND[0]) / DBOUND[2])  # 118


# ----------------------------- GT discretization ---------------------------
def test_binning_known_depths():
    img, feat = (4, 4), (2, 2)  # downsample factor 2
    gt = torch.zeros(1, 1, 4, 4)
    gt[0, 0, 0, 0] = 1.0    # patch (0,0) -> bin floor((1.0-1)/0.5) = 0
    gt[0, 0, 2, 2] = 10.0   # patch (1,1) -> bin floor((10-1)/0.5) = 18
    one_hot, valid = depth_sup.downsample_gt_depth(gt, img, feat, DBOUND, NUM_BINS)
    assert one_hot.shape == (1, 2, 2, NUM_BINS)
    assert valid.shape == (1, 2, 2)
    assert valid[0, 0, 0] and one_hot[0, 0, 0].argmax().item() == 0
    assert valid[0, 1, 1] and one_hot[0, 1, 1].argmax().item() == 18
    # patches without any LiDAR point are NOT valid (not supervised)
    assert not valid[0, 0, 1]
    assert not valid[0, 1, 0]


def test_min_pool_keeps_nearest_nonzero():
    img, feat = (2, 2), (1, 1)  # one patch of 4 pixels
    gt = torch.tensor([[[[5.0, 0.0], [0.0, 3.0]]]])  # min non-zero = 3.0
    one_hot, valid = depth_sup.downsample_gt_depth(gt, img, feat, DBOUND, NUM_BINS)
    assert valid[0, 0, 0]
    # 3.0 -> bin floor((3-1)/0.5) = 4 (zeros must NOT win the min)
    assert one_hot[0, 0, 0].argmax().item() == 4


# ------------------------------- sparse mask --------------------------------
def test_empty_gt_gives_zero_loss():
    img, feat = (4, 4), (2, 2)
    logits = torch.randn(1, NUM_BINS, 2, 2, requires_grad=True)
    gt = torch.zeros(1, 1, 4, 4)  # no LiDAR points anywhere
    loss = depth_sup.depth_bce_loss(logits, gt, img, feat, DBOUND, NUM_BINS, 0.5)
    assert float(loss) == 0.0
    loss.backward()  # graph stays alive, zero gradient


def test_only_valid_pixels_are_supervised():
    img, feat = (4, 4), (2, 2)
    gt = torch.zeros(1, 1, 4, 4)
    gt[0, 0, 0, 0] = 2.0  # exactly one valid feature cell (0,0)
    one_hot, _ = depth_sup.downsample_gt_depth(gt, img, feat, DBOUND, NUM_BINS)
    b = one_hot[0, 0, 0].argmax().item()
    logits = torch.full((1, NUM_BINS, 2, 2), -10.0)
    logits[0, b, 0, 0] = 10.0  # confident & correct at the only valid cell
    loss = depth_sup.depth_bce_loss(logits, gt, img, feat, DBOUND, NUM_BINS, 1.0)
    assert float(loss) < 0.1
    # changing logits at an INVALID cell must not change the loss at all
    logits2 = logits.clone()
    logits2[0, :, 1, 1] = 999.0
    loss2 = depth_sup.depth_bce_loss(logits2, gt, img, feat, DBOUND, NUM_BINS, 1.0)
    assert torch.allclose(loss, loss2)


# ------------------- softmax + BCE (activation-matched) ---------------------
def test_uses_softmax_then_bce():
    # [module-A/fix-softmax-bce] loss must equal BCE on the softmax-over-bins
    # probabilities (matching the forward LSS lift), NOT per-bin BCE-with-logits.
    # [module-A/fix-loss-norm] normalization is official BEVDepth's: per-pixel
    # SUM over the D bins / n_valid_pixels (here n_valid = 1), NOT mean over
    # n_valid * D elements.
    img, feat, D = (2, 2), (1, 1), 4
    gt = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])  # depth 1.0 -> bin 0
    logits = torch.tensor([[[[3.0]], [[-1.0]], [[0.0]], [[2.0]]]])  # (1,4,1,1)
    loss = depth_sup.depth_bce_loss(logits, gt, img, feat, DBOUND, D, 1.0)
    flat = logits.permute(0, 2, 3, 1).reshape(1, D).float()
    tgt = torch.zeros(1, D)
    tgt[0, 0] = 1.0
    ref_softmax = F.binary_cross_entropy(
        flat.softmax(1), tgt, reduction='none').sum() / 1
    assert torch.allclose(loss, ref_softmax)          # softmax(dim=bins) + BCE
    ref_mean = F.binary_cross_entropy(flat.softmax(1), tgt, reduction='mean')
    assert not torch.allclose(loss, ref_mean)         # NOT meaned over bins too
    ref_logits = F.binary_cross_entropy_with_logits(flat, tgt, reduction='none')
    assert not torch.allclose(loss, ref_logits.sum())  # NOT per-bin logits-BCE


def test_official_bevdepth_normalization():
    # [module-A/fix-loss-norm] with n_valid pixels, loss = weight *
    # sum-over-(pixels, bins) BCE / n_valid — the exact official BEVDepth
    # reduction (base_exp.py: `.sum() / max(1.0, fg_mask.sum())`, `3.0 *`).
    img, feat = (4, 4), (2, 2)
    gt = torch.zeros(1, 1, 4, 4)
    gt[0, 0, 0, 0] = 2.0   # valid cell (0,0)
    gt[0, 0, 2, 2] = 30.0  # valid cell (1,1)  -> n_valid = 2
    torch.manual_seed(0)
    logits = torch.randn(1, NUM_BINS, 2, 2)
    loss = depth_sup.depth_bce_loss(logits, gt, img, feat, DBOUND, NUM_BINS, 3.0)
    one_hot, valid = depth_sup.downsample_gt_depth(gt, img, feat, DBOUND, NUM_BINS)
    assert int(valid.sum()) == 2
    probs = logits.float().softmax(dim=1).permute(0, 2, 3, 1)[valid]
    ref = F.binary_cross_entropy(
        probs, one_hot[valid], reduction='none').sum() / valid.sum()
    assert torch.allclose(loss, 3.0 * ref)


def test_dummy_forward_no_nan_and_softmax_dim():
    # full-size dummy: (B, D=118, 32, 88) logits + sparse GT at 256x704 (ds=8)
    B, D, fH, fW = 2, NUM_BINS, 32, 88
    img, feat = (256, 704), (fH, fW)
    torch.manual_seed(0)
    logits = torch.randn(B, D, fH, fW, requires_grad=True)
    gt = torch.zeros(B, 1, *img)
    gt[0, 0, 10, 20] = 7.3       # -> a valid bin
    gt[0, 0, 100, 300] = 25.0
    gt[1, 0, 200, 600] = 41.1
    loss = depth_sup.depth_bce_loss(logits, gt, img, feat, DBOUND, D, 0.5)
    assert torch.isfinite(loss)                       # no NaN/Inf
    loss.backward()                                   # differentiable
    assert torch.isfinite(logits.grad).all()
    # softmax is over the depth-bin dim (dim=1): per-pixel probs sum to 1
    s = logits.detach().softmax(dim=1).sum(dim=1)     # (B, fH, fW)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-4)


def test_weight_scales_loss():
    img, feat = (2, 2), (1, 1)
    gt = torch.tensor([[[[5.0, 0.0], [0.0, 0.0]]]])
    logits = torch.randn(1, NUM_BINS, 1, 1)
    l1 = depth_sup.depth_bce_loss(logits, gt, img, feat, DBOUND, NUM_BINS, 1.0)
    l05 = depth_sup.depth_bce_loss(logits, gt, img, feat, DBOUND, NUM_BINS, 0.5)
    assert torch.allclose(l05, 0.5 * l1)


# ------------- input dropout (v2, closes the leakage shortcut) ---------------
def test_drop_keep_ratio_one_is_noop():
    # [module-A/depth-sup-v2] keep_ratio >= 1.0 returns the SAME tensor (v1).
    d = torch.rand(4, 1, 16, 16)
    out = depth_sup.drop_depth_input(d, 1.0)
    assert out is d  # identity: no copy, byte-identical to v1 input path


def test_drop_never_mutates_input_so_gt_stays_full():
    # the caller stashes `d` as the FULL GT then calls drop on the same tensor;
    # drop must NOT mutate it, or the supervision target would be corrupted.
    d = torch.rand(2, 1, 32, 32)
    gt_ref = d.clone()
    _ = depth_sup.drop_depth_input(d, 0.3)
    assert torch.equal(d, gt_ref)  # input untouched -> GT unaffected


def test_drop_zeros_stay_zero_and_rate_is_right():
    # only real points (non-zero) can be dropped; empty pixels stay empty.
    torch.manual_seed(0)
    d = torch.ones(200, 1, 40, 40)          # all "points" so we can measure rate
    d[:, :, ::2, :] = 0.0                    # half are empty (stay empty)
    out = depth_sup.drop_depth_input(d, 0.3)
    # empty pixels are never resurrected
    assert torch.all(out[d == 0.0] == 0.0)
    # kept fraction of the real points is ~0.3
    real = d > 0.0
    kept = (out[real] > 0.0).float().mean().item()
    assert abs(kept - 0.3) < 0.02
    # kept values are unchanged (masked multiply by 1), not rescaled
    assert torch.all((out[real] == 1.0) | (out[real] == 0.0))


# ------------- DepthLSSTransform level (needs compiled ops) -----------------
def _load_depth_lss():
    try:
        from projects.BEVFusion.bevfusion.depth_lss import DepthLSSTransform
        return DepthLSSTransform
    except Exception:
        return None


_DepthLSS = _load_depth_lss()


def _make_vt(use_depth_sup, keep_ratio=1.0):
    return _DepthLSS(
        in_channels=8, out_channels=4, image_size=[16, 16], feature_size=[2, 2],
        xbound=[-54.0, 54.0, 0.3], ybound=[-54.0, 54.0, 0.3],
        zbound=[-10.0, 10.0, 20.0], dbound=DBOUND, downsample=1,
        use_depth_sup=use_depth_sup, depth_loss_weight=0.5,
        depth_input_keep_ratio=keep_ratio)


@pytest.mark.skipif(_DepthLSS is None,
                    reason='DepthLSSTransform needs the compiled BEVFusion ops')
def test_caches_only_when_enabled():
    x = torch.randn(1, 1, 8, 2, 2)
    d = torch.zeros(1, 1, 1, 16, 16)
    d[0, 0, 0, 0, 0] = 5.0

    vt_off = _make_vt(False)
    vt_off.get_cam_feats(x.clone(), d.clone())
    assert vt_off._depth_pred_logits is None and vt_off._depth_gt is None

    vt_on = _make_vt(True)
    vt_on.get_cam_feats(x.clone(), d.clone())
    assert vt_on._depth_pred_logits is not None
    assert vt_on._depth_pred_logits.shape == (1, vt_on.D, 2, 2)
    assert vt_on._depth_gt is not None and vt_on._depth_gt.shape == (1, 1, 16, 16)
    # stashed values are PRE-softmax logits (their softmax sums to 1; raw doesn't)
    probs = vt_on._depth_pred_logits.softmax(1)
    assert torch.allclose(probs.sum(1), torch.ones_like(probs.sum(1)), atol=1e-4)
    # get_depth_loss runs, clears caches, and is differentiable
    loss = vt_on.get_depth_loss()
    assert vt_on._depth_pred_logits is None and vt_on._depth_gt is None
    loss.backward()


@pytest.mark.skipif(_DepthLSS is None,
                    reason='DepthLSSTransform needs the compiled BEVFusion ops')
def test_no_new_parameters_vs_baseline():
    on = {n for n, _ in _make_vt(True).named_parameters()}
    off = {n for n, _ in _make_vt(False).named_parameters()}
    assert on == off  # depth supervision introduces zero new parameters
    # v2 input dropout also adds zero parameters (it is a runtime mask)
    v2 = {n for n, _ in _make_vt(True, keep_ratio=0.3).named_parameters()}
    assert v2 == off


@pytest.mark.skipif(_DepthLSS is None,
                    reason='DepthLSSTransform needs the compiled BEVFusion ops')
def test_v2_gt_is_full_while_training_input_is_dropped():
    # [module-A/depth-sup-v2] with keep_ratio<1 in TRAIN mode, the stashed GT
    # keeps ALL points (full projection) even though the depthnet input is
    # dropped. We assert the GT nonzero count equals the original full map.
    torch.manual_seed(0)
    d = torch.zeros(1, 1, 1, 16, 16)
    pts = [(2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (1, 1)]
    for (r, c) in pts:
        d[0, 0, 0, r, c] = 5.0 + r  # distinct in-range depths
    n_full = int((d > 0).sum())

    vt = _make_vt(True, keep_ratio=0.25).train()
    vt.get_cam_feats(torch.randn(1, 1, 8, 2, 2), d.clone())
    # GT keeps the FULL set of points (supervision target never dropped)
    assert int((vt._depth_gt > 0).sum()) == n_full


@pytest.mark.skipif(_DepthLSS is None,
                    reason='DepthLSSTransform needs the compiled BEVFusion ops')
def test_v2_dropout_is_train_only():
    # in EVAL mode the input is NOT dropped regardless of keep_ratio, so the
    # produced camera features are identical to keep_ratio=1.0 (inference
    # unchanged). We compare get_cam_feats outputs with a fixed seed.
    torch.manual_seed(0)
    d = torch.zeros(1, 1, 1, 16, 16)
    d[0, 0, 0, 4, 4] = 7.0
    d[0, 0, 0, 8, 8] = 20.0
    x = torch.randn(1, 1, 8, 2, 2)

    vt = _make_vt(True, keep_ratio=0.1).eval()
    out_drop = vt.get_cam_feats(x.clone(), d.clone())
    vt_ref = _make_vt(True, keep_ratio=1.0).eval()
    vt_ref.load_state_dict(vt.state_dict())  # same weights
    out_full = vt_ref.get_cam_feats(x.clone(), d.clone())
    assert torch.allclose(out_drop, out_full)  # eval: dropout is a no-op


def test_invalid_keep_ratio_raises():
    if _DepthLSS is None:
        pytest.skip('DepthLSSTransform needs the compiled BEVFusion ops')
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(AssertionError):
            _make_vt(True, keep_ratio=bad)
