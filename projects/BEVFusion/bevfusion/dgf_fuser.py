# Copyright (c) OpenMMLab. All rights reserved.
#
# DepthFusion: Depth-Aware Hybrid Feature Fusion for LiDAR-Camera 3D Object
# Detection (arXiv:2505.07398), Section III-B "Depth-GFusion (DGF)".
#
# This module implements ONLY DGF (the *global* depth-guided BEV fusion).
# It deliberately does NOT implement DLF (Depth-LFusion, Sec. III-C), whose
# instance-level local re-fusion overlaps with the InsFusion module (D) and
# would make the ablation redundant.
#
# `DGFFuser` is a drop-in replacement for the baseline `ConvFuser`
# (projects/BEVFusion/bevfusion/transfusion_head.py): identical call interface
# `forward(inputs=[img_bev, lidar_bev]) -> fused_bev`, switched purely via the
# config field `model.fusion_layer.type`. When the config keeps `ConvFuser`,
# the baseline code path is byte-identical (this file is never imported into
# the forward graph). Hence the "+C off == baseline" guarantee is structural.
#
# Points the paper leaves unspecified are implemented with the values listed
# in projects/BEVFusion/IMPL_NOTES_C.md and tagged `ASSUMPTION (A#)` below.
import math
from contextlib import nullcontext
from typing import List, Optional

import torch
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from torch import nn

from mmdet3d.registry import MODELS


def _sinusoidal_embedding(values: torch.Tensor,
                          dim: int,
                          temperature: float = 10000.0) -> torch.Tensor:
    """Embed a scalar field into ``dim`` channels via sin/cos (no parameters).

    Transformer-style sinusoidal embedding (Vaswani et al., DepthFusion ref
    [28]) applied element-wise to a (H, W) scalar map.

    Args:
        values (Tensor): scalar field of shape (H, W).
        dim (int): number of output channels, must be even.
        temperature (float): frequency base.

    Returns:
        Tensor: (dim, H, W) sinusoidal embedding, ``requires_grad=False``.
    """
    assert dim % 2 == 0, 'embedding dim must be even'
    H, W = values.shape
    half = dim // 2
    freq = torch.arange(half, dtype=torch.float32, device=values.device)
    freq = temperature**(-freq / half)                          # (half,)
    ang = values.reshape(1, H, W) * freq.reshape(half, 1, 1)     # (half,H,W)
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=0)    # (dim,H,W)


def build_pos_encoding(H: int,
                       W: int,
                       dim: int,
                       device: torch.device,
                       temperature: float = 10000.0) -> torch.Tensor:
    """2D sinusoidal positional encoding ``P`` (DETR-style), no parameters.

    Half the channels encode the row (y) index, half the column (x) index.
    DepthFusion Sec. III-B: ``P`` is added element-wise to the original BEV
    features. ASSUMPTION (A3): the paper only says "positional encoding"; we
    use a parameter-free 2D sine PE (ref [28]).
    """
    ys = torch.arange(H, dtype=torch.float32, device=device).reshape(H, 1).expand(H, W)
    xs = torch.arange(W, dtype=torch.float32, device=device).reshape(1, W).expand(H, W)
    pe_y = _sinusoidal_embedding(ys, dim // 2, temperature)
    pe_x = _sinusoidal_embedding(xs, dim // 2, temperature)
    return torch.cat([pe_y, pe_x], dim=0)                        # (dim,H,W)


def build_depth_encoding(H: int,
                         W: int,
                         dim: int,
                         device: torch.device,
                         temperature: float = 10000.0) -> torch.Tensor:
    """Depth encoding ``D`` — DepthFusion Eq.(1)/(2), no parameters.

    Eq.(1): each BEV cell ``p_k = {(x_k, y_k) : d_k}``.
    Eq.(2): ``d_k = E((x_k, y_k), (x_{n/2}, y_{n/2}))`` — the Euclidean
    distance from cell (x_k, y_k) to the ego-centre cell (H//2, W//2).
    The depth matrix ``M`` is then turned into the depth encoding ``D`` by
    applying sin/cos (DepthFusion Sec. III-B). ASSUMPTION (A4): distance is
    measured in BEV-cell-index units.
    """
    cy, cx = H // 2, W // 2
    ys = torch.arange(H, dtype=torch.float32, device=device).reshape(H, 1)
    xs = torch.arange(W, dtype=torch.float32, device=device).reshape(1, W)
    dist = torch.sqrt((ys - cy)**2 + (xs - cx)**2)               # M, (H,W)
    return _sinusoidal_embedding(dist, dim, temperature)         # (dim,H,W)


def efficient_sdpa_ctx():
    """Context manager forcing the memory-efficient / flash SDPA backend and
    DISABLING the math backend.

    The DGF global cross-attention runs over the full BEV grid (180x180 =
    32400 tokens). The math backend would materialise an O(N^2) attention
    matrix (~32400^2 per head) and OOM a 24GB card. Forcing flash /
    mem-efficient keeps memory O(N); disabling math makes an unavailable
    kernel raise loudly instead of silently falling back and OOM-ing.

    Version-robust: torch>=2.1 exposes ``torch.nn.attention.sdpa_kernel``;
    torch 2.0.x (the pinned training env) uses
    ``torch.backends.cuda.sdp_kernel``. See IMPL_NOTES_C.md (A7) for how to
    confirm the chosen backend at runtime.
    """
    try:  # torch >= 2.1
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel([SDPBackend.FLASH_ATTENTION,
                            SDPBackend.EFFICIENT_ATTENTION])
    except Exception:  # torch 2.0.x
        from torch.backends.cuda import sdp_kernel
        return sdp_kernel(
            enable_flash=True, enable_mem_efficient=True, enable_math=False)


@MODELS.register_module()
class DGFFuser(nn.Module):
    """Depth-GFusion fuser — DepthFusion (arXiv:2505.07398) Sec. III-B.

    Drop-in replacement for ``ConvFuser``. Globally fuses the LiDAR BEV
    (query, modulated by the depth encoding D) with the image BEV (key/value)
    via a depth-aware multi-head cross-attention (Eq.3), followed by a
    residual conv-FFN aggregation (Eq.4).

    Args:
        in_channels (list[int]): ``[img_bev_channels, lidar_bev_channels]``,
            e.g. ``[80, 256]`` (mirrors the baseline ConvFuser interface).
        out_channels (int): output channels; must equal ``embed_dims`` (no
            output reshape), fed to ``pts_backbone`` (256).
        embed_dims (int): common attention dim ``C``. Default 256.
            ASSUMPTION (A1): paper uses 128; we use 256 to match the baseline
            and avoid an extra output projection (A13).
        num_heads (int): attention heads. Default 8 -> head_dim 32 (A5).
        norm_cfg (dict): normalization in Eq.(4). Default ``BN2d`` (becomes
            SyncBN under ``--sync_bn torch``, consistent with ConvFuser/
            baseline). Switch to ``dict(type='GN', num_groups=32)`` if small-
            batch training is unstable (A8).
        ffn_channels (int | None): conv-FFN hidden channels. Default
            ``embed_dims`` (A9).
        use_out_proj (bool): add a 1x1 output projection W_O after attention.
            Default ``False`` (A2b).
        pe_temperature / depth_temperature (float): sin/cos frequency bases.
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: int = 256,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 norm_cfg: Optional[dict] = None,
                 ffn_channels: Optional[int] = None,
                 use_out_proj: bool = False,
                 pe_temperature: float = 10000.0,
                 depth_temperature: float = 10000.0) -> None:
        super().__init__()
        assert len(in_channels) == 2, \
            'in_channels must be [img_bev_ch, lidar_bev_ch]'
        assert out_channels == embed_dims, \
            'DGFFuser keeps out_channels == embed_dims (no output reshape)'
        assert embed_dims % num_heads == 0, \
            'embed_dims must be divisible by num_heads'
        img_ch, lidar_ch = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        self.pe_temperature = pe_temperature
        self.depth_temperature = depth_temperature
        if norm_cfg is None:
            norm_cfg = dict(type='BN2d')  # A8

        # --- channel-align projections (DepthFusion Sec. III-B) ---
        # ASSUMPTION (A2): these 1x1 convs ARE the attention projections:
        #   W_Q = lidar_proj ; W_K = W_V = img_proj (key/value share img_proj).
        # No separate per-head QKV linears.
        self.lidar_proj = nn.Conv2d(lidar_ch, embed_dims, kernel_size=1)
        self.img_proj = nn.Conv2d(img_ch, embed_dims, kernel_size=1)
        # ASSUMPTION (A2b): optional output projection W_O, default off.
        self.out_proj = nn.Conv2d(embed_dims, embed_dims,
                                  kernel_size=1) if use_out_proj else None

        # --- Eq.(4) aggregation: two norms + conv-FFN ---
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        ffn_channels = ffn_channels or embed_dims  # A9
        self.ffn = nn.Sequential(
            nn.Conv2d(embed_dims, ffn_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ffn_channels, embed_dims, kernel_size=3, padding=1),
        )

        # Caches for P and D. Stored as plain (non-parameter, non-buffer)
        # tensors so they never appear in `parameters()` (D/P are param-free,
        # A3/A4) nor bloat the `state_dict`. Rebuilt lazily if (H, W) or device
        # changes (A12: resolution-agnostic).
        self._pe = None
        self._de = None
        self._cached_hw = None

    def _get_pe_de(self, H, W, device, dtype):
        if (self._cached_hw != (H, W) or self._pe is None
                or self._pe.device != device):
            self._pe = build_pos_encoding(H, W, self.embed_dims, device,
                                          self.pe_temperature)
            self._de = build_depth_encoding(H, W, self.embed_dims, device,
                                            self.depth_temperature)
            self._cached_hw = (H, W)
        return self._pe.to(dtype), self._de.to(dtype)

    def _cross_attention(self, q, k, v, B, H, W):
        """Multi-head cross-attention via SDPA (DepthFusion Eq.3).

        Inputs are (B, C, H, W); reshaped to (B, heads, H*W, head_dim) and fed
        to ``scaled_dot_product_attention`` (scaling 1/sqrt(head_dim), which
        deviates from the paper's 1/sqrt(C) -- ASSUMPTION A6). On CUDA the call
        is wrapped to use the flash / mem-efficient backend only.
        """
        def to_heads(x):
            # (B,C,H,W) -> (B, heads, head_dim, H*W) -> (B, heads, H*W, head_dim)
            return x.reshape(B, self.num_heads, self.head_dim,
                             H * W).permute(0, 1, 3, 2).contiguous()

        qh, kh, vh = to_heads(q), to_heads(k), to_heads(v)
        ctx = efficient_sdpa_ctx() if q.is_cuda else nullcontext()
        with ctx:
            out = F.scaled_dot_product_attention(qh, kh, vh)  # (B,heads,HW,hd)
        # (B,heads,HW,hd) -> (B, C, H, W)
        return out.permute(0, 1, 3, 2).reshape(B, self.embed_dims, H, W)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # `inputs` follows the baseline ConvFuser order: [img_bev, lidar_bev]
        # (see bevfusion.py extract_feat: features = [img_feature, pts_feature]).
        img_bev, lidar_bev = inputs[0], inputs[1]
        assert img_bev.shape[-2:] == lidar_bev.shape[-2:], (
            'img/lidar BEV must share spatial size (as the baseline fusion '
            f'requires): {img_bev.shape[-2:]} vs {lidar_bev.shape[-2:]}')
        B, _, H, W = lidar_bev.shape

        # (1) channel alignment to the common dim C (DepthFusion Sec. III-B)
        v_gb = self.lidar_proj(lidar_bev)   # V_GB (query stream), (B,C,H,W)
        i_gb = self.img_proj(img_bev)       # I_GB (key/value stream), (B,C,H,W)

        # positional encoding P and depth encoding D (param-free)
        pe, de = self._get_pe_de(H, W, lidar_bev.device, v_gb.dtype)
        pe = pe.unsqueeze(0)                # (1,C,H,W)
        de = de.unsqueeze(0)               # (1,C,H,W)

        # (2)/(3) DepthFusion Eq.(3): depth-modulated cross-attention
        #   query = (V_GB + P) ⊙ D ; key = I_GB + P ; value = I_GB
        q = (v_gb + pe) * de
        k = i_gb + pe
        v = i_gb
        v_hat = self._cross_attention(q, k, v, B, H, W)  # V̂_GB
        if self.out_proj is not None:
            v_hat = self.out_proj(v_hat)

        # (5) DepthFusion Eq.(4): F_GB = N(FFN(N(V̂+V_GB)) + N(V̂+V_GB))
        #   ASSUMPTION (A10): aggregation in (B,C,H,W) layout (FFN = convs);
        #   ASSUMPTION (A12): residual base is V_GB (lidar_proj) -> lidar-centric.
        x = self.norm1(v_hat + v_gb)
        out = self.norm2(self.ffn(x) + x)
        return out
