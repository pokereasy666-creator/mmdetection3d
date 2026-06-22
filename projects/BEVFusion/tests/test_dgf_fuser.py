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


def test_residual_stabilization_init_and_groupnorm():
    # [module-C/fix-residual-stability] out_proj is always-on + ZERO-init, gamma
    # (ReZero) inits 0.05, and the output is non-negative (out_relu) like
    # ConvFuser. Also exercises the configurable GroupNorm.
    fuser = DGFFuser(in_channels=[80, 256], embed_dims=256, num_heads=8,
                     norm_cfg=dict(type='GN', num_groups=32))
    assert fuser.out_proj is not None
    assert float(fuser.out_proj.weight.abs().sum()) == 0.0   # zero-init weight
    assert float(fuser.out_proj.bias.abs().sum()) == 0.0     # zero-init bias
    assert torch.allclose(fuser.gamma.detach(),
                          torch.tensor([0.05]))              # ReZero init 0.05
    out = fuser([torch.randn(1, 80, 16, 16), torch.randn(1, 256, 16, 16)])
    assert out.shape == (1, 256, 16, 16)
    assert float(out.min()) >= 0.0   # out_relu -> non-negative, like ConvFuser


def test_default_norm_is_groupnorm():
    # [module-C/bn-to-groupnorm] The default DGF norm is GroupNorm, NOT BN2d:
    # BatchNorm renormalised the sparse, low-variance LiDAR feature (std~0.056)
    # to ~1, amplifying the norm ~18x -> NaN gradients. GroupNorm is
    # batch-independent and robust to sparse features.
    fuser = DGFFuser(in_channels=[80, 256], embed_dims=256, num_heads=8)
    assert isinstance(fuser.norm1, torch.nn.GroupNorm)
    assert isinstance(fuser.norm2, torch.nn.GroupNorm)
    assert fuser.norm1.num_groups == 32
    assert fuser.norm1.num_channels == 256
    # GroupNorm carries no BatchNorm running stats
    assert not any('running_mean' in k or 'running_var' in k
                   for k in fuser.state_dict())


def test_attn_dim_decoupled_from_output():
    # [module-C/dgf-attn-dim] attn_dim (attention internal width) is decoupled
    # from embed_dims/out_channels (output, must stay 256). With attn_dim=128:
    #   img_proj/out_proj/q_proj/P/D run at 128, but the output stays 256.
    fuser = DGFFuser(in_channels=[80, 256], embed_dims=256, out_channels=256,
                     attn_dim=128, num_heads=4)
    assert fuser.attn_dim == 128 and fuser.embed_dims == 256
    assert fuser.head_dim == 32                      # 128 / 4 heads
    assert isinstance(fuser.q_proj, torch.nn.Conv2d)  # real down-proj when 128!=256
    assert fuser.q_proj.out_channels == 128
    assert fuser.img_proj.out_channels == 128
    assert fuser.out_proj.in_channels == 128 and fuser.out_proj.out_channels == 256
    assert float(fuser.out_proj.weight.abs().sum()) == 0.0  # still zero-init
    out = fuser([torch.randn(1, 80, 16, 16), torch.randn(1, 256, 16, 16)])
    assert out.shape == (1, 256, 16, 16)              # output dim preserved
    assert float(out.min()) >= 0.0                    # out_relu -> non-negative


def test_attn_dim_defaults_to_embed_dims():
    # attn_dim=None -> equals embed_dims; q_proj is Identity (old behaviour).
    fuser = DGFFuser(in_channels=[80, 256], embed_dims=256, num_heads=8)
    assert fuser.attn_dim == 256
    assert isinstance(fuser.q_proj, torch.nn.Identity)
    assert fuser.img_proj.out_channels == 256
    assert fuser.out_proj.in_channels == 256 and fuser.out_proj.out_channels == 256


def test_invalid_args():
    with pytest.raises(AssertionError):
        DGFFuser(in_channels=[80, 256], out_channels=128, embed_dims=256)
    with pytest.raises(AssertionError):
        DGFFuser(in_channels=[80, 256], embed_dims=256, num_heads=7)
