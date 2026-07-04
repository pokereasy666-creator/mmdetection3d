# Copyright (c) OpenMMLab. All rights reserved.
"""Faithful channel-adapted Depth-GFusion for BEVFusion."""

import os
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmengine.model import BaseModule
from torch import nn

from mmdet3d.registry import MODELS


def _sinusoidal_encoding(values: torch.Tensor,
                         embed_dims: int,
                         dtype: torch.dtype,
                         temperature: float = 10000.0) -> torch.Tensor:
    """Encode a scalar field with parameter-free sine/cosine channels."""
    if embed_dims % 2 != 0:
        raise ValueError('The sinusoidal encoding dimension must be even.')

    num_frequencies = embed_dims // 2
    frequencies = torch.arange(
        num_frequencies, device=values.device, dtype=torch.float32)
    frequencies = temperature**(-frequencies / num_frequencies)
    phases = values.float().unsqueeze(0) * frequencies[:, None, None]
    encoding = torch.cat((phases.sin(), phases.cos()), dim=0)
    return encoding.to(dtype=dtype)


def build_position_encoding(
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
        embed_dims: int = 128) -> torch.Tensor:
    """Build a 2D sine/cosine position encoding of shape (1, C, H, W)."""
    if embed_dims % 4 != 0:
        raise ValueError(
            'embed_dims must be divisible by 4 for 2D position encoding.')

    y_coords = torch.arange(
        height, device=device, dtype=torch.float32)[:, None].expand(
            height, width)
    x_coords = torch.arange(
        width, device=device, dtype=torch.float32)[None, :].expand(
            height, width)
    y_encoding = _sinusoidal_encoding(y_coords, embed_dims // 2, dtype)
    x_encoding = _sinusoidal_encoding(x_coords, embed_dims // 2, dtype)
    return torch.cat((y_encoding, x_encoding), dim=0).unsqueeze(0)


def build_depth_encoding(
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
        embed_dims: int = 128) -> torch.Tensor:
    """Build the parameter-free radial BEV depth encoding (1, C, H, W)."""
    center_y = (height - 1) / 2
    center_x = (width - 1) / 2
    y_coords = torch.arange(
        height, device=device, dtype=torch.float32)[:, None]
    x_coords = torch.arange(
        width, device=device, dtype=torch.float32)[None, :]
    depth = torch.sqrt((x_coords - center_x)**2 +
                       (y_coords - center_y)**2)
    return _sinusoidal_encoding(depth, embed_dims, dtype).unsqueeze(0)


@MODELS.register_module()
class DGFFuserV1(BaseModule):
    """DepthFusion DGF adapted through a 128-dimensional attention bottleneck.

    The input order matches BEVFusion's fusion call:
    ``inputs=[img_bev, lidar_bev]``.
    """

    def __init__(
        self,
        in_channels: Sequence[int] = (80, 256),
        embed_dims: int = 128,
        out_channels: int = 256,
        num_heads: int = 8,
        norm_cfg: dict = dict(type='GN', num_groups=32),
        ffn_channels: int = 128,
        zero_init_out_proj: bool = False,
        init_cfg: dict = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        if len(in_channels) != 2:
            raise ValueError(
                'in_channels must contain [image_channels, lidar_channels].')
        if embed_dims % num_heads != 0:
            raise ValueError('embed_dims must be divisible by num_heads.')
        if embed_dims % 4 != 0:
            raise ValueError(
                'embed_dims must be divisible by 4 for position encoding.')
        if out_channels != in_channels[1]:
            raise ValueError(
                'out_channels must equal the LiDAR channels for residual add.')
        if ffn_channels <= 0:
            raise ValueError('ffn_channels must be positive.')

        norm_cfg = norm_cfg.copy()
        if norm_cfg.get('type') == 'GN':
            num_groups = norm_cfg.get('num_groups')
            if not isinstance(num_groups, int) or num_groups <= 0:
                raise ValueError(
                    'GN norm_cfg must provide a positive num_groups.')
            if embed_dims % num_groups != 0:
                raise ValueError(
                    'embed_dims must be divisible by GN num_groups.')

        img_channels, lidar_channels = in_channels
        self.in_channels = tuple(in_channels)
        self.embed_dims = embed_dims
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads

        self.lidar_q_proj = nn.Conv2d(lidar_channels, embed_dims, 1)
        self.img_k_proj = nn.Conv2d(img_channels, embed_dims, 1)
        self.img_v_proj = nn.Conv2d(img_channels, embed_dims, 1)
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = nn.Sequential(
            nn.Conv2d(embed_dims, ffn_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ffn_channels, embed_dims, 3, padding=1),
        )
        self.out_proj = nn.Conv2d(embed_dims, out_channels, 1)
        if zero_init_out_proj:
            nn.init.zeros_(self.out_proj.weight)
            if self.out_proj.bias is not None:
                nn.init.zeros_(self.out_proj.bias)

        # Plain attributes: excluded from parameters, buffers, and state_dict.
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

        query_heads = to_heads(query)
        key_heads = to_heads(key)
        value_heads = to_heads(value)
        attended = F.scaled_dot_product_attention(
            query_heads,
            key_heads,
            value_heads,
            dropout_p=0.0,
            is_causal=False)
        return attended.permute(0, 1, 3, 2).contiguous().reshape(
            batch_size, self.embed_dims, height, width).contiguous()

    @staticmethod
    def _is_rank_zero() -> bool:
        return (not torch.distributed.is_available()
                or not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0)

    def _debug_log(self, lidar_128: torch.Tensor, img_k_128: torch.Tensor,
                   img_v_128: torch.Tensor, attn_out: torch.Tensor,
                   delta_256: torch.Tensor,
                   lidar_bev: torch.Tensor) -> None:
        def stats(name: str, tensor: torch.Tensor) -> str:
            tensor_float = tensor.detach().float()
            return (
                f'{name} mean={tensor_float.mean().item():.5f} '
                f'std={tensor_float.std().item():.5f} '
                f'finite={bool(torch.isfinite(tensor_float).all())}')

        delta_norm = delta_256.detach().float().norm()
        lidar_norm = lidar_bev.detach().float().norm()
        ratio = (delta_norm / lidar_norm.clamp_min(1e-12)).item()
        print(
            f'[DGFV1-DEBUG] call={self._debug_forward_count} | '
            f'{stats("lidar_128", lidar_128)} | '
            f'{stats("img_k_128", img_k_128)} | '
            f'{stats("img_v_128", img_v_128)} | '
            f'{stats("attn_out", attn_out)} | '
            f'{stats("delta_256", delta_256)} | '
            f'|delta_256|/|lidar_bev|={ratio:.6f}',
            flush=True)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        if len(inputs) != 2:
            raise ValueError('DGFFuserV1 expects [img_bev, lidar_bev].')
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

        lidar_128 = self.lidar_q_proj(lidar_bev)
        img_k_128 = self.img_k_proj(img_bev)
        img_v_128 = self.img_v_proj(img_bev)
        if (img_k_128.device != lidar_128.device
                or img_v_128.device != lidar_128.device):
            raise ValueError(
                'Projected image and LiDAR features must share a device.')
        if (img_k_128.dtype != lidar_128.dtype
                or img_v_128.dtype != lidar_128.dtype):
            raise ValueError(
                'Projected image and LiDAR features must share a dtype.')

        _, _, height, width = lidar_128.shape
        position, depth = self._get_encodings(
            height, width, lidar_128)
        query = (lidar_128 + position) * depth
        key = img_k_128 + position
        value = img_v_128

        attn_out = self._cross_attention(query, key, value)
        fused = self.norm1(attn_out + lidar_128)
        fused = self.norm2(self.ffn(fused) + fused)
        delta_256 = self.out_proj(fused)
        out = lidar_bev + delta_256

        self._debug_forward_count += 1
        if (os.environ.get('DGFV1_DEBUG') == '1'
                and self._is_rank_zero()
                and (self._debug_forward_count == 1
                     or self._debug_forward_count % 50 == 0)):
            self._debug_log(lidar_128, img_k_128, img_v_128, attn_out,
                            delta_256, lidar_bev)

        return out.contiguous()
