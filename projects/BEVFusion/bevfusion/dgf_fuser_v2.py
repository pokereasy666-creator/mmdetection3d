# Copyright (c) OpenMMLab. All rights reserved.
"""Paper-exact Depth-GFusion (DepthFusion arXiv:2505.07398, Eq. 1-4).

Unlike ``DGFFuserV1`` (the 128-dim channel-adapted variant with an extra
outer residual), this module follows the paper's structure as literally as
the BEVFusion baseline allows:

- fusion runs at the LiDAR trunk width (256): the LiDAR BEV feature enters
  Eq. 3/4 natively, with NO learned projection or compression;
- the image BEV feature is adapted 80 -> 256 by a single 1x1 conv (the one
  unavoidable deviation: the paper has both modalities at the same C) and
  this SAME tensor serves as both key and value, exactly as Eq. 3 uses one
  I_B^G for K and V;
- there are no learned Q/K/V/output projections inside the attention -
  Eq. 3 contains none; "multi-head" is a plain channel-group split;
- Add & Norm uses per-position LayerNorm (the transformer reading of N());
- the module output IS Eq. 4's F. No outer residual, no out_proj.
"""

import os
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
from mmengine.model import BaseModule
from torch import nn
from torch.utils.checkpoint import checkpoint

from mmdet3d.registry import MODELS

from .dgf_fuser_v1 import build_depth_encoding, build_position_encoding


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel dim of an NCHW map (per-position)."""

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


@MODELS.register_module()
class DGFFuserV2(BaseModule):
    """Paper-exact DGF at the native LiDAR trunk width.

    The input order matches BEVFusion's fusion call:
    ``inputs=[img_bev, lidar_bev]``. The output width equals the LiDAR
    channel count (``in_channels[1]``), feeding pts_backbone unchanged.

    Args:
        in_channels: ``[image_channels, lidar_channels]``; fusion runs at
            ``lidar_channels``.
        num_heads: channel-group count for the multi-head split.
        ffn_channels: hidden width of the two-conv FFN (Eq. 4).
        use_checkpoint: recompute the fusion block in backward instead of
            storing activations. Memory escape hatch for OOM, flippable at
            launch via --cfg-options; does not change the math.
    """

    def __init__(
        self,
        in_channels: Sequence[int] = (80, 256),
        num_heads: int = 8,
        ffn_channels: int = 256,
        use_checkpoint: bool = False,
        init_cfg: dict = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        if len(in_channels) != 2:
            raise ValueError(
                'in_channels must contain [image_channels, lidar_channels].')
        img_channels, lidar_channels = in_channels
        if lidar_channels % num_heads != 0:
            raise ValueError(
                'lidar_channels must be divisible by num_heads.')
        if lidar_channels % 4 != 0:
            raise ValueError(
                'lidar_channels must be divisible by 4 for the 2D position '
                'encoding.')
        if ffn_channels <= 0:
            raise ValueError('ffn_channels must be positive.')

        self.in_channels = tuple(in_channels)
        self.embed_dims = lidar_channels
        self.num_heads = num_heads
        self.head_dim = lidar_channels // num_heads
        self.use_checkpoint = use_checkpoint

        # The single unavoidable channel adapter (paper: both BEVs share C).
        self.img_proj = nn.Conv2d(img_channels, lidar_channels, 1)
        self.norm1 = ChannelLayerNorm(lidar_channels)
        self.norm2 = ChannelLayerNorm(lidar_channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(lidar_channels, ffn_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ffn_channels, lidar_channels, 3, padding=1),
        )

        # Plain attributes: excluded from parameters, buffers, state_dict.
        self._encoding_cache_key = None
        self._cached_position_encoding = None
        self._cached_depth_encoding = None
        self._debug_forward_count = 0

    def _get_encodings(
            self, height: int, width: int, reference: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = reference.device
        dtype = reference.dtype
        cache_key = (height, width, device.type, device.index, dtype)
        if self._encoding_cache_key != cache_key:
            self._cached_position_encoding = build_position_encoding(
                height, width, device, dtype, self.embed_dims)
            self._cached_depth_encoding = build_depth_encoding(
                height, width, device, dtype, self.embed_dims)
            self._encoding_cache_key = cache_key
        return (self._cached_position_encoding,
                self._cached_depth_encoding)

    def _cross_attention(self, query: torch.Tensor, key: torch.Tensor,
                         value: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = query.shape

        def to_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(
                batch_size, self.num_heads, self.head_dim,
                height * width).permute(0, 1, 3, 2).contiguous()

        attended = F.scaled_dot_product_attention(
            to_heads(query),
            to_heads(key),
            to_heads(value),
            dropout_p=0.0,
            is_causal=False)
        return attended.permute(0, 1, 3, 2).contiguous().reshape(
            batch_size, self.embed_dims, height, width).contiguous()

    def _fuse(self, lidar_bev: torch.Tensor,
              img_bev_c: torch.Tensor) -> torch.Tensor:
        """Eq. 3 + Eq. 4 at the native trunk width."""
        _, _, height, width = lidar_bev.shape
        position, depth = self._get_encodings(height, width, lidar_bev)
        query = (lidar_bev + position) * depth
        key = img_bev_c + position
        value = img_bev_c

        attn_out = self._cross_attention(query, key, value)
        hidden = self.norm1(attn_out + lidar_bev)
        return self.norm2(self.ffn(hidden) + hidden)

    @staticmethod
    def _is_rank_zero() -> bool:
        return (not torch.distributed.is_available()
                or not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0)

    def _debug_log(self, lidar_bev: torch.Tensor,
                   out: torch.Tensor) -> None:
        out_float = out.detach().float()
        lidar_float = lidar_bev.detach().float()
        delta_ratio = ((out_float - lidar_float).norm() /
                       lidar_float.norm().clamp_min(1e-12)).item()
        print(
            f'[DGFV2-DEBUG] call={self._debug_forward_count} | '
            f'out mean={out_float.mean().item():.5f} '
            f'std={out_float.std().item():.5f} '
            f'finite={bool(torch.isfinite(out_float).all())} | '
            f'|out-lidar|/|lidar|={delta_ratio:.6f}',
            flush=True)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        if len(inputs) != 2:
            raise ValueError('DGFFuserV2 expects [img_bev, lidar_bev].')
        img_bev, lidar_bev = inputs
        if img_bev.ndim != 4 or lidar_bev.ndim != 4:
            raise ValueError('Both BEV inputs must be 4D NCHW tensors.')
        if img_bev.shape[1] != self.in_channels[0]:
            raise ValueError(
                f'Expected {self.in_channels[0]} image channels, '
                f'but received {img_bev.shape[1]}.')
        if lidar_bev.shape[1] != self.in_channels[1]:
            raise ValueError(
                f'Expected {self.in_channels[1]} LiDAR channels, '
                f'but received {lidar_bev.shape[1]}.')
        if img_bev.shape[0] != lidar_bev.shape[0]:
            raise ValueError('Image and LiDAR batch sizes must match.')
        if img_bev.shape[-2:] != lidar_bev.shape[-2:]:
            raise ValueError('Image and LiDAR BEV spatial sizes must match.')

        img_bev_c = self.img_proj(img_bev)
        if self.use_checkpoint and torch.is_grad_enabled():
            out = checkpoint(
                self._fuse, lidar_bev, img_bev_c, use_reentrant=False)
        else:
            out = self._fuse(lidar_bev, img_bev_c)

        self._debug_forward_count += 1
        if (os.environ.get('DGFV2_DEBUG') == '1'
                and self._is_rank_zero()
                and (self._debug_forward_count == 1
                     or self._debug_forward_count % 50 == 0)):
            self._debug_log(lidar_bev, out)

        return out.contiguous()
