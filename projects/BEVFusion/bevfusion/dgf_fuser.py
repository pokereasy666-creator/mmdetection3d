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
import os
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


class LayerNorm2d(nn.LayerNorm):
    """Transformer Add&Norm over the channel (token-feature) dim for NCHW.

    DepthFusion Eq.(4) ``N`` is the transformer "Add & Norm" (LayerNorm). Here
    each BEV cell is a token and its C-dim feature vector is normalised
    independently: for every ``(b, h, w)`` location we normalise over ``C``.
    This is **per-sample**, carries **no batch statistics** and needs **no
    cross-GPU sync** -- it removes the SyncBN path that caused the fp16 nan at
    full 180 (BN renormalised the sparse, low-variance LiDAR BEV std~0.056 -> 1,
    a ~18x norm blow-up). It also symmetrically bounds ``V̂ + V_B``, which is
    why the faithful module needs no camera gate (gamma).

    KNOWN RISK (validated in smoke, not assumed away): on sparse LiDAR many
    empty cells are ~constant vectors (the ``lidar_proj`` bias); LayerNorm maps
    a constant vector to its affine ``bias``, i.e. it lifts empty background
    from ~0 to a nonzero constant and can compress fg/bg contrast -- the LN
    analogue of the BN blow-up. Mitigation ladder if smoke flags it: LN with
    ``bias`` forced 0 -> GroupNorm -> full-sample LN (see IMPL_NOTES_C A8).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)  # (B,C,H,W) -> (B,H,W,C)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias,
                         self.eps)
        return x.permute(0, 3, 1, 2)  # back to (B,C,H,W)


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
        attn_resolution (int | None): if set, the global attention runs on a
            downsampled ``attn_resolution`` x ``attn_resolution`` BEV (A14) and
            the camera increment is bilinearly upsampled back; the LiDAR residual
            base / aggregation / output stay full-res. Default ``None`` -> full
            resolution (byte-identical). E.g. 135 (180->135) cuts N ~1.78x =>
            ~3.16x cheaper attention fwd+bwd; lower values are faster but coarser.
        norm_cfg (dict | None): normalization layer ``N`` in Eq.(4). Default
            ``None`` -> channel-wise ``LayerNorm2d`` (A8): the transformer
            "Add & Norm", per-sample, no batch stats, no cross-GPU sync. This is
            the faithful choice and removes the SyncBN+fp16 nan path. Pass an
            explicit cfg (e.g. ``dict(type='GN', num_groups=32)`` or
            ``dict(type='BN2d')``) to override for experiments.
        ffn_channels (int | None): conv-FFN hidden channels. Default
            ``embed_dims`` (A9).
        pe_temperature / depth_temperature (float): sin/cos frequency bases.

    Faithful aggregation (DepthFusion Eq.4): ``U = N(V̂_GB + V_GB)`` then
    ``F_GB = N(FFN(U) + U)`` -- the camera increment ``V̂_GB`` and the LiDAR
    base ``V_GB`` are added **1:1** (no ReZero ``gamma``, no zero-init
    ``out_proj``, no final ``ReLU``). ``LayerNorm`` symmetrically bounds the sum,
    so the camera branch cannot explode and needs no gate; it participates from
    step 0.
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: int = 256,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 attn_resolution: Optional[int] = None,
                 norm_cfg: Optional[dict] = None,
                 ffn_channels: Optional[int] = None,
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
        # [module-C/dgf-attn-downsample] optional spatial downsampling of the
        # attention grid: run the global cross-attention on an
        # attn_resolution x attn_resolution BEV (cuts N -> N^2 fwd+bwd), then
        # bilinearly upsample the camera increment back. None -> full resolution
        # (byte-identical). The LiDAR residual base / output stay full-res.
        assert attn_resolution is None or attn_resolution >= 1, \
            'attn_resolution must be a positive int or None'
        self.attn_resolution = attn_resolution
        self.pe_temperature = pe_temperature
        self.depth_temperature = depth_temperature
        self.norm_cfg = norm_cfg  # A8: None -> faithful channel-wise LayerNorm

        # --- channel-align projections (DepthFusion Sec. III-B) ---
        # ASSUMPTION (A2): these 1x1 convs ARE the attention projections:
        #   W_Q = lidar_proj ; W_K = W_V = img_proj (key/value share img_proj).
        # No separate per-head QKV linears.
        self.lidar_proj = nn.Conv2d(lidar_ch, embed_dims, kernel_size=1)
        self.img_proj = nn.Conv2d(img_ch, embed_dims, kernel_size=1)
        # A2b: multi-head attention output projection W_O (1x1). DEFAULT init
        # (no zero-init): the camera increment is live from step 0 -- the module
        # is faithful and does NOT suppress the camera branch.
        self.out_proj = nn.Conv2d(embed_dims, embed_dims, kernel_size=1)

        # --- Eq.(4) aggregation: two norms + conv-FFN ---
        # A8: default N = channel-wise LayerNorm (LayerNorm2d). An explicit
        # norm_cfg (GN/BN2d) still overrides via build_norm_layer.
        if norm_cfg is None:
            self.norm1 = LayerNorm2d(embed_dims)
            self.norm2 = LayerNorm2d(embed_dims)
        else:
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
        self._dbg_calls = 0  # [dgf-debug] forward counter for env-guarded prints
        self._nan_reported = False  # [dgf-nan] localizer fires once per run
        # [dgf-perf] env-guarded (DGF_PERF=1) counters/timers; default zero impact.
        self._pe_builds = 0   # times P/D were actually (re)built -> proves cache
        self._attn_calls = 0  # SDPA calls -> proves one attention per forward

    def _get_pe_de(self, H, W, device, dtype):
        if (self._cached_hw != (H, W) or self._pe is None
                or self._pe.device != device):
            self._pe = build_pos_encoding(H, W, self.embed_dims, device,
                                          self.pe_temperature)
            self._de = build_depth_encoding(H, W, self.embed_dims, device,
                                            self.depth_temperature)
            self._cached_hw = (H, W)
            self._pe_builds += 1  # [dgf-perf] real (re)build counter
            if os.environ.get('DGF_PERF') == '1':
                print(f'[DGF-PERF] built P/D #{self._pe_builds} (H,W)=({H},{W}) '
                      f'device={device} (should print ONCE for the whole run)',
                      flush=True)
        return self._pe.to(dtype), self._de.to(dtype)

    def _cross_attention(self, q, k, v, B, H, W):
        """Multi-head cross-attention via SDPA (DepthFusion Eq.3).

        Inputs are (B, C, H, W); reshaped to (B, heads, H*W, head_dim) and fed
        to ``scaled_dot_product_attention``. Head config (A5/A6): C=256 -> 8
        heads -> head_dim=32 -> scaling 1/sqrt(32). Eq.(3) writes 1/sqrt(C);
        in the multi-head realisation the "real head dim" (head_dim) is the
        correct per-head scale, self-consistent at C=256. On CUDA the call is
        wrapped to use the flash / mem-efficient backend only.
        """
        self._attn_calls += 1  # [dgf-perf] proves one SDPA call per forward

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
        v_gb = self.lidar_proj(lidar_bev)   # V_GB residual base (FULL res), (B,C,H,W)
        i_gb = self.img_proj(img_bev)       # I_GB (key/value stream), (B,C,H,W)

        # [module-C/dgf-attn-downsample] run the global attention on a coarser
        # (Hl,Wl) grid to cut N (=> N^2 attention fwd+bwd). The residual base
        # v_gb stays FULL resolution; only the attention inputs are average-pooled,
        # and the camera increment v_hat is bilinearly upsampled back before the
        # residual add. attn_resolution=None (or >= H,W) -> full-res (no-op).
        R = self.attn_resolution
        if R is not None and (R < H or R < W):
            Hl, Wl = min(R, H), min(R, W)
            v_gb_a = F.adaptive_avg_pool2d(v_gb, (Hl, Wl))
            i_gb_a = F.adaptive_avg_pool2d(i_gb, (Hl, Wl))
            do_up = True
        else:
            Hl, Wl, v_gb_a, i_gb_a, do_up = H, W, v_gb, i_gb, False

        # positional encoding P and depth encoding D (param-free), built at the
        # attention grid (Hl, Wl)
        pe, de = self._get_pe_de(Hl, Wl, lidar_bev.device, v_gb.dtype)
        pe = pe.unsqueeze(0)                # (1,C,Hl,Wl)
        de = de.unsqueeze(0)               # (1,C,Hl,Wl)

        # (2)/(3) DepthFusion Eq.(3): depth-modulated cross-attention
        #   query = (V_GB + P) ⊙ D ; key = I_GB + P ; value = I_GB
        q = (v_gb_a + pe) * de
        k = i_gb_a + pe
        v = i_gb_a
        # [dgf-perf] env-guarded forward timing: attention vs the rest (aggregation).
        # CUDA events measure the FORWARD only; the attention backward runs later in
        # loss.backward() and is inferred from the overall step-time delta.
        _perf = os.environ.get('DGF_PERF') == '1' and q.is_cuda
        if _perf:
            _e0 = torch.cuda.Event(enable_timing=True)
            _e1 = torch.cuda.Event(enable_timing=True)
            _e0.record()
        v_hat = self._cross_attention(q, k, v, B, Hl, Wl)  # V̂_GB at (Hl,Wl)
        if _perf:
            _e1.record()
        # [dgf-debug/bug2-probe] name the pre-out_proj attention output (softmax·V)
        # so the DGF_DEBUG block can read |attn_out| -- numerically identical to
        # the previous `v_hat = self.out_proj(v_hat)` (pure renaming, no behavior
        # change). `attn_out` is at the (Hl,Wl) attention grid (= full res when
        # attn_resolution is None).
        attn_out = v_hat
        v_hat = self.out_proj(attn_out)  # W_O (default init): camera live from step 0
        if do_up:  # upsample the camera increment back to full res
            v_hat = F.interpolate(v_hat, size=(H, W), mode='bilinear',
                                  align_corners=False)

        # (5) DepthFusion Eq.(4): U = N(V̂_GB + V_GB) ; F_GB = N(FFN(U) + U).
        #   A10: aggregation in (B,C,H,W) layout (FFN = convs);
        #   A12: residual base is V_GB (lidar_proj) -> lidar-centric, 256 ch.
        # V̂_GB and V_GB are added 1:1 -- NO gamma gate, NO out_relu. LayerNorm
        # (norm1/norm2) symmetrically bounds the sum so the camera cannot
        # explode. Intermediates are NAMED (not re-computed) so the DGF_DEBUG
        # block below reads per-step stats without re-calling norm1/norm2/ffn.
        u = self.norm1(v_gb + v_hat)
        ffn_out = self.ffn(u)
        out = self.norm2(ffn_out + u)

        if _perf:
            _e2 = torch.cuda.Event(enable_timing=True)
            _e2.record()
            torch.cuda.synchronize()
            _rank0 = (not torch.distributed.is_available()
                      or not torch.distributed.is_initialized()
                      or torch.distributed.get_rank() == 0)
            _every = int(os.environ.get('DGF_PERF_EVERY', '50'))
            if _rank0 and self._attn_calls % _every == 1:
                print(
                    f'[DGF-PERF] fwd attn={_e0.elapsed_time(_e1):.1f}ms '
                    f'aggregation(out_proj+norm+ffn)={_e1.elapsed_time(_e2):.1f}ms '
                    f'| attn_calls={self._attn_calls} pe_builds={self._pe_builds}',
                    flush=True)

        # [dgf-debug] env-guarded diagnostics (set DGF_DEBUG=1). No effect on the
        # forward result, params, or normal/no-env runs. Two parts:
        #   (a) nan/inf LOCALIZER -- every step, cheap; reports the FIRST
        #       non-finite tensor (fwd) and the FIRST non-finite GRAD (bwd, via
        #       hooks). It only LOCALIZES -- no suppression.
        #   (b) periodic STATS (every DGF_DEBUG_EVERY) -- attention collapse,
        #       camera-contribution probe, and the LayerNorm sparse-background
        #       observables.
        if os.environ.get('DGF_DEBUG') == '1':
            self._dbg_calls += 1
            every = int(os.environ.get('DGF_DEBUG_EVERY', '50'))
            rank0 = (not torch.distributed.is_available()
                     or not torch.distributed.is_initialized()
                     or torch.distributed.get_rank() == 0)
            if rank0 and self._dbg_calls == 1:
                print(
                    '[DGF-DEBUG] sdpa '
                    f'flash={torch.backends.cuda.flash_sdp_enabled()} '
                    f'mem_efficient={torch.backends.cuda.mem_efficient_sdp_enabled()} '
                    f'math={torch.backends.cuda.math_sdp_enabled()}',
                    flush=True)

            # (a) nan/inf localizer. Scan fwd tensors in compute order; the
            # first non-finite one is where it first breaks. q/k proxy the
            # attention logits; v_hat = attention output; u/ffn_out/out =
            # aggregation. Reported once per run (self._nan_reported) to avoid
            # spam; backward hooks catch grad nan/inf in reverse order.
            named = [('q', q), ('k', k), ('v_hat', v_hat),
                     ('u', u), ('ffn_out', ffn_out), ('out', out)]
            if not self._nan_reported:
                for nm, t in named:
                    if not torch.isfinite(t.detach()).all():
                        self._nan_reported = True
                        if rank0:
                            print(f'[DGF-NAN] forward: FIRST non-finite tensor '
                                  f'= {nm} @ call={self._dbg_calls} '
                                  f'shape={tuple(t.shape)}', flush=True)
                        break
            if not self._nan_reported:
                def _mk_hook(nm, call):
                    def _hook(g):
                        if (g is not None and not self._nan_reported
                                and not torch.isfinite(g).all()):
                            self._nan_reported = True
                            if rank0:
                                print(f'[DGF-NAN] backward: FIRST non-finite '
                                      f'GRAD = {nm} @ call={call}', flush=True)
                    return _hook
                for nm, t in named:
                    if t.requires_grad:
                        t.register_hook(_mk_hook(nm, self._dbg_calls))

            # (b) periodic stats.
            if rank0 and self._dbg_calls % every == 1:
                with torch.no_grad():
                    base = v_gb.norm().item()
                    incr = v_hat.norm().item()

                    # q/k live on the (Hl,Wl) attention grid, not (H,W)
                    Nl = Hl * Wl

                    def _th(t):
                        return t.reshape(B, self.num_heads, self.head_dim,
                                         Nl).permute(0, 1, 3, 2)

                    qh, kh = _th(q.float()), _th(k.float())
                    s = min(64, Nl)
                    idx = torch.randperm(Nl, device=q.device)[:s]
                    attn = ((qh[:, :, idx, :] @ kh.transpose(-2, -1))
                            / (self.head_dim**0.5)).softmax(-1)
                    ent = -(attn * (attn + 1e-12).log()).sum(-1).mean().item()
                    maxp = attn.max(-1).values.mean().item()

                    def _stat(name, t):
                        tf = t.float()
                        return (f'{name}: norm={tf.norm().item():.3f} '
                                f'mean={tf.mean().item():.4f} '
                                f'std={tf.std().item():.4f} '
                                f'nan={bool(torch.isnan(tf).any())} '
                                f'inf={bool(torch.isinf(tf).any())}')

                    print(
                        f'[DGF-DEBUG step-norms] call={self._dbg_calls} | '
                        + ' | '.join([
                            _stat('1.v_gb(in)', v_gb),
                            _stat('2.u=norm1(v_gb+vhat)', u),
                            _stat('3.ffn_out', ffn_out),
                            _stat('4.out=norm2(ffn+u)', out),
                        ]),
                        flush=True)

                    # Camera-contribution probe. F_lidar is a COUNTERFACTUAL
                    # reference recomputed with v_hat:=0 -- it is NOT a path the
                    # model uses and does NOT mean "camera off at step 0" (the
                    # camera is live from step 1; gamma is gone). It is purely a
                    # yardstick for how far the camera moves the fused output:
                    # rel_cam = ||F - F_lidar|| / ||F_lidar||, want stably >0.05.
                    # Skipped for BatchNorm overrides (re-calling BN in train
                    # mode would double-update its running stats).
                    rel_cam = float('nan')
                    if not isinstance(self.norm1,
                                      (nn.BatchNorm2d, nn.SyncBatchNorm)):
                        u_l = self.norm1(v_gb)
                        f_lidar = self.norm2(self.ffn(u_l) + u_l)
                        rel_cam = ((out - f_lidar).norm()
                                   / max(f_lidar.norm().item(), 1e-6)).item()

                    # LayerNorm sparse-background observables (the LN-axis risk):
                    # empty cells = raw LiDAR BEV cells with ~0 channel-vector.
                    # Report their share, the post-norm magnitude THERE, and the
                    # fg/bg contrast before (v_gb) vs after (u) norm1 -- a big
                    # drop = LN compressed background contrast.
                    cell = lidar_bev.float().norm(dim=1)          # (B,H,W)
                    empty = cell < 1e-6
                    er = empty.float().mean().item()
                    vgb_c = v_gb.float().norm(dim=1)
                    u_c = u.float().norm(dim=1)

                    def _mm(t, m):
                        return t[m].mean().item() if bool(m.any()) else float('nan')

                    bg_pre, fg_pre = _mm(vgb_c, empty), _mm(vgb_c, ~empty)
                    bg_post, fg_post = _mm(u_c, empty), _mm(u_c, ~empty)
                    c_pre = fg_pre / max(bg_pre, 1e-6)
                    c_post = fg_post / max(bg_post, 1e-6)

                    print(
                        f'[DGF-DEBUG] call={self._dbg_calls} '
                        f'|vhat(cam)|={incr:.3f} |vgb(lidar)|={base:.3f} '
                        f'raw_ratio={incr / max(base, 1e-6):.4f} | '
                        f'rel_cam(||F-Flidar||/||Flidar||)={rel_cam:.4f} | '
                        f'attn_entropy={ent:.3f} (uniform={math.log(Nl):.3f}) '
                        f'max_prob={maxp:.4f} | '
                        f'out mean={out.mean().item():.3f} '
                        f'std={out.std().item():.3f} norm={out.norm().item():.3f}',
                        flush=True)
                    print(
                        f'[DGF-DEBUG ln-sparse] call={self._dbg_calls} '
                        f'empty_ratio={er:.3f} '
                        f'bg|post-norm1|={bg_post:.3f} fg|post-norm1|={fg_post:.3f} '
                        f'contrast_pre(fg/bg)={c_pre:.3f} '
                        f'contrast_post(fg/bg)={c_post:.3f}',
                        flush=True)

                    # [bug2-probe] Stage-A localization of the |vhat|/|vgb|
                    # 7-132x blow-up. Measures (does NOT fix) the three candidate
                    # sources so the decision table can be applied:
                    #   - |i_gb| vs |vgb| : is the image-BEV value already huge at
                    #     DGF input? (|i_gb|>>|vgb| -> upstream view-transform)
                    #   - max-cell|i_gb|  : a few huge image cells the attention
                    #     could broadcast across all N BEV cells?
                    #   - |attn_out|      : softmax·V BEFORE out_proj. If
                    #     |attn_out|~=|vhat|>>|i_gb| -> amplification is INSIDE
                    #     the attention ("已爆在 attention 内").
                    #   - ||W_O||         : TRAINED out_proj weight norm (init
                    #     argument can't rule out a grown W_O). >>O(1) -> out_proj.
                    igb = i_gb.float()
                    igb_norm = igb.norm().item()
                    igb_maxcell = igb.norm(dim=1).max().item()
                    attn_out_norm = attn_out.float().norm().item()
                    wo_norm = self.out_proj.weight.float().norm().item()
                    print(
                        f'[DGF-DEBUG bug2] call={self._dbg_calls} '
                        f'|i_gb(img-val)|={igb_norm:.3f} '
                        f'max-cell|i_gb|={igb_maxcell:.3f} | '
                        f'|attn_out(softmax.V,pre-Wo)|={attn_out_norm:.3f} '
                        f'|vhat(post-Wo)|={incr:.3f} | '
                        f'||W_O||={wo_norm:.4f} | '
                        f'ref |vgb(lidar)|={base:.3f}',
                        flush=True)

                    # [bug1-preview] GN(v_gb) contrast look-ahead (nice-to-have,
                    # zero footprint): manual GroupNorm (32 groups, weight=1/
                    # bias=0, axis-only) on v_gb -> does GN preserve fg/bg
                    # contrast (unlike per-cell LN, which forces it to 1.000)?
                    # This previews BUG 1's fix WITHOUT changing the live norm1.
                    G = 32
                    vg = v_gb.float()
                    Bv, Cv, Hv, Wv = vg.shape
                    gg = vg.reshape(Bv, G, Cv // G, Hv, Wv)
                    gmean = gg.mean(dim=(2, 3, 4), keepdim=True)
                    gvar = gg.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
                    gn = ((gg - gmean) / (gvar + 1e-5).sqrt()).reshape(
                        Bv, Cv, Hv, Wv)
                    gn_c = gn.norm(dim=1)
                    gn_bg, gn_fg = _mm(gn_c, empty), _mm(gn_c, ~empty)
                    gn_contrast = gn_fg / max(gn_bg, 1e-6)
                    print(
                        f'[DGF-DEBUG bug1-preview gn(v_gb)] call={self._dbg_calls} '
                        f'bg|gn|={gn_bg:.3f} fg|gn|={gn_fg:.3f} '
                        f'contrast_gn(fg/bg)={gn_contrast:.3f} '
                        f'(LN forces 1.000; GN should be >1)',
                        flush=True)

        # Return a contiguous NCHW tensor (like ConvFuser's Conv2d output): the
        # attention permute/reshape + mixed-memory-format residual can leave a
        # non-standard stride / channels-last layout that Conv2d preserves all
        # the way to TransFusionHead, where `heatmap.view(B, -1)` then fails
        # ("view size is not compatible ... use .reshape"). Forcing contiguity
        # here keeps the downstream contract identical to the baseline fuser.
        return out.contiguous()
