# Copyright (c) OpenMMLab. All rights reserved.
# [module-C/DGF] CPU unit tests for DGFFuser (DepthFusion arXiv:2505.07398).
#
# Runs on CPU with small fake tensors; needs only torch + mmcv + mmdet3d
# (NO compiled CUDA ops, NO GPU). We import `dgf_fuser.py` *standalone* (via
# sys.path) instead of `projects.BEVFusion.bevfusion`, because the package
# __init__ pulls in bevfusion.py -> ops/*.so which require the compiled
# extensions. On CPU, SDPA transparently uses the math backend (fine for the
# 16x16 = 256-token shapes here).
#
# Run from the repo root, e.g.:
#   pytest projects/BEVFusion/tests/test_dgf_fuser.py -q
import os
import sys

import pytest
import torch

# import dgf_fuser.py directly, bypassing the project package __init__
_BEVFUSION_DIR = os.path.join(os.path.dirname(__file__), '..', 'bevfusion')
sys.path.insert(0, os.path.abspath(_BEVFUSION_DIR))
import dgf_fuser  # noqa: E402

DGFFuser = dgf_fuser.DGFFuser


def _fake_inputs(B=2, H=16, W=16):
    img_bev = torch.randn(B, 80, H, W, requires_grad=True)
    lidar_bev = torch.randn(B, 256, H, W, requires_grad=True)
    return img_bev, lidar_bev


def test_output_shape():
    fuser = DGFFuser(in_channels=[80, 256], out_channels=256,
                     embed_dims=256, num_heads=8)
    img_bev, lidar_bev = _fake_inputs()
    out = fuser([img_bev, lidar_bev])
    assert out.shape == (2, 256, 16, 16)


def test_pe_de_have_no_trainable_params():
    """Depth encoding D and positional encoding P must be parameter-free
    (DepthFusion Sec. III-B: D is obtained without introducing parameters)."""
    fuser = DGFFuser(in_channels=[80, 256], embed_dims=256, num_heads=8)
    # trigger lazy construction of P/D
    pe, de = fuser._get_pe_de(16, 16, torch.device('cpu'), torch.float32)
    assert pe.requires_grad is False
    assert de.requires_grad is False
    param_names = [n for n, _ in fuser.named_parameters()]
    # P/D are stored as plain attributes -> never appear as parameters/buffers
    assert all('_pe' not in n and '_de' not in n for n in param_names)
    assert '_pe' not in dict(fuser.state_dict())
    assert '_de' not in dict(fuser.state_dict())
    # only the projections / FFN / norms carry trainable params
    assert any('lidar_proj' in n for n in param_names)
    assert any('img_proj' in n for n in param_names)
    assert any('ffn' in n for n in param_names)


def test_backward_runs():
    fuser = DGFFuser(in_channels=[80, 256], embed_dims=256, num_heads=8)
    img_bev, lidar_bev = _fake_inputs()
    out = fuser([img_bev, lidar_bev])
    loss = out.float().sum()
    loss.backward()
    assert img_bev.grad is not None
    assert lidar_bev.grad is not None
    # projection weights receive gradients
    assert fuser.lidar_proj.weight.grad is not None
    assert fuser.img_proj.weight.grad is not None


def test_resolution_agnostic():
    """P/D are rebuilt for the actual (H, W); a different grid still works."""
    fuser = DGFFuser(in_channels=[80, 256], embed_dims=256, num_heads=8)
    out1 = fuser([torch.randn(1, 80, 16, 16), torch.randn(1, 256, 16, 16)])
    out2 = fuser([torch.randn(1, 80, 20, 24), torch.randn(1, 256, 20, 24)])
    assert out1.shape == (1, 256, 16, 16)
    assert out2.shape == (1, 256, 20, 24)


def test_faithful_no_suppression_stack():
    # Faithful DGF (DepthFusion Eq.4) has NO camera-suppression stack: no ReZero
    # gamma, no zero-init out_proj, no final ReLU. out_proj is a normal W_O
    # (default init -> nonzero), and the norm-ended output may be negative.
    fuser = DGFFuser(in_channels=[80, 256], embed_dims=256, num_heads=8)
    assert not hasattr(fuser, 'gamma')          # ReZero gate removed
    assert not hasattr(fuser, 'out_relu')       # final ReLU removed
    assert fuser.out_proj is not None
    assert float(fuser.out_proj.weight.abs().sum()) > 0.0   # NOT zero-init
    out = fuser([torch.randn(1, 80, 16, 16), torch.randn(1, 256, 16, 16)])
    assert out.shape == (1, 256, 16, 16)
    # faithful output is GroupNorm-ended -> signed (no out_relu clamp)
    assert float(out.min()) < 0.0


def test_camera_contributes():
    # The camera branch must move the fused output (no suppression). Compare the
    # real output F against the counterfactual v_hat=0 reference F_lidar; they
    # must differ. (F_lidar is a yardstick, NOT a path the model uses.)
    torch.manual_seed(0)
    fuser = DGFFuser(in_channels=[80, 256], embed_dims=256, num_heads=8).eval()
    img = torch.randn(1, 80, 16, 16)
    lidar = torch.randn(1, 256, 16, 16)
    with torch.no_grad():
        out = fuser([img, lidar])
        # zero the image stream -> camera increment goes to (near) zero
        out_zero_cam = fuser([torch.zeros_like(img), lidar])
    rel = (out - out_zero_cam).norm() / out_zero_cam.norm().clamp_min(1e-6)
    assert float(rel) > 0.05   # camera meaningfully changes the output


def test_default_norm_is_groupnorm():
    # Default norm N = GroupNorm(32) [A8], NOT channel-wise LayerNorm (rejected
    # by the 850-step smoke: per-cell LN forces fg/bg contrast to 1.000). GN has
    # no batch running stats / no cross-GPU sync. BN remains an explicit override.
    fuser = DGFFuser(in_channels=[80, 256], embed_dims=256, num_heads=8)
    assert isinstance(fuser.norm1, torch.nn.GroupNorm)
    assert isinstance(fuser.norm2, torch.nn.GroupNorm)
    assert fuser.norm1.num_groups == 32
    assert fuser.norm1.num_channels == 256
    # no BatchNorm running stats anywhere
    assert not any('running_mean' in k or 'running_var' in k
                   for k in fuser.state_dict())
    # explicit override still works
    bn = DGFFuser(in_channels=[80, 256], embed_dims=256, num_heads=8,
                  norm_cfg=dict(type='BN2d'))
    assert isinstance(bn.norm1, torch.nn.BatchNorm2d)


def test_groupnorm_preserves_contrast():
    # BUG-1 regression guard: GroupNorm (stats shared across space) must NOT
    # equalize a high-magnitude (foreground) cell with a low-magnitude (empty)
    # cell -- that fg/bg contrast is exactly what per-cell LayerNorm destroyed
    # (forcing every cell to norm sqrt(C)). Build a feature map with one bright
    # column and one ~zero column; after GroupNorm the bright column's per-cell
    # norm must stay clearly larger than the empty column's.
    gn = torch.nn.GroupNorm(32, 256)
    x = torch.zeros(1, 256, 8, 8)
    x[:, :, :, 0] = 5.0          # bright (foreground) column
    # column 1 stays ~0 (empty/background)
    with torch.no_grad():
        y = gn(x)
    cell_norm = y.norm(dim=1)    # (1,8,8)
    fg = cell_norm[:, :, 0].mean()
    bg = cell_norm[:, :, 1].mean()
    assert float(fg / bg.clamp_min(1e-6)) > 1.5   # contrast preserved (>1)


def test_attn_resolution_downsamples_attention():
    # [module-C/dgf-attn-downsample] with attn_resolution set, the attention runs
    # on a downsampled grid but the output stays at the input resolution and
    # non-negative; channels stay 256 (embed_dims/num_heads unchanged).
    fuser = DGFFuser(in_channels=[80, 256], embed_dims=256, num_heads=8,
                     attn_resolution=8)
    assert fuser.attn_resolution == 8
    assert fuser.head_dim == 32                       # 256 / 8 heads (unchanged)
    out = fuser([torch.randn(1, 80, 16, 16), torch.randn(1, 256, 16, 16)])
    assert out.shape == (1, 256, 16, 16)              # full-res output preserved
    # P/D were built on the 8x8 attention grid, not 16x16
    assert fuser._cached_hw == (8, 8)


def test_attn_resolution_none_is_full_res():
    # attn_resolution=None (default) -> attention on the full grid (no pooling).
    fuser = DGFFuser(in_channels=[80, 256], embed_dims=256, num_heads=8)
    assert fuser.attn_resolution is None
    out = fuser([torch.randn(1, 80, 16, 16), torch.randn(1, 256, 16, 16)])
    assert out.shape == (1, 256, 16, 16)
    assert fuser._cached_hw == (16, 16)               # built on full grid


def test_invalid_args():
    with pytest.raises(AssertionError):
        DGFFuser(in_channels=[80, 256], out_channels=128, embed_dims=256)
    with pytest.raises(AssertionError):
        DGFFuser(in_channels=[80, 256], embed_dims=256, num_heads=7)
