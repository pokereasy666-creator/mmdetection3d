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


def test_use_out_proj_and_groupnorm():
    fuser = DGFFuser(in_channels=[80, 256], embed_dims=256, num_heads=8,
                     use_out_proj=True,
                     norm_cfg=dict(type='GN', num_groups=32))
    assert fuser.out_proj is not None
    out = fuser([torch.randn(1, 80, 16, 16), torch.randn(1, 256, 16, 16)])
    assert out.shape == (1, 256, 16, 16)


def test_invalid_args():
    with pytest.raises(AssertionError):
        DGFFuser(in_channels=[80, 256], out_channels=128, embed_dims=256)
    with pytest.raises(AssertionError):
        DGFFuser(in_channels=[80, 256], embed_dims=256, num_heads=7)
