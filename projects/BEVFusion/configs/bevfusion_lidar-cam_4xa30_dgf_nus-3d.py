# [module-C/DGF] +C variant: BEVFusion LiDAR-camera with DepthFusion's
# Depth-GFusion (DGF) replacing ConvFuser as the BEV fusion layer.
#   DepthFusion (arXiv:2505.07398) Sec. III-B.
#
# Inherits the verified 4xA30 baseline config UNCHANGED (batch_size 2,
# accumulative_counts 4 -> effective batch 32, AMP loss_scale=512.0, SyncBN,
# cyclic-20e, lr 2e-4) and ONLY swaps `model.fusion_layer` to `DGFFuser`.
# This keeps the +C variant trained under the exact same hardware/optimisation
# setup as the baseline (fairness rule). Toggling baseline<->+C is one field.
#
# Offline-deployment notes (see RUNBOOK_C.md):
#   - work_dir is set OUTSIDE the source tree (placeholder below) so runtime
#     products survive re-extracting the source zip. REPLACE <FILL_ME>.
#   - pretrained weights are injected at launch via --cfg-options with LOCAL
#     absolute paths (no online URLs). No download happens from this config.

_base_ = ['./bevfusion_lidar-cam_voxel0075_4xa30-amp-accum_nus-3d.py']

# Replace ConvFuser with DGFFuser. `_delete_=True` drops the inherited
# ConvFuser dict entirely so no stale keys (e.g. out_channels-only) leak in.
model = dict(
    fusion_layer=dict(
        _delete_=True,
        type='DGFFuser',
        in_channels=[80, 256],   # [img_bev_ch, lidar_bev_ch]
        out_channels=256,        # == embed_dims; feeds pts_backbone(in=256)
        embed_dims=256,          # aggregation/OUTPUT dim == out_channels (A1/A13)
        attn_dim=128,            # [module-C/dgf-attn-dim] attention internal width
                                 #   (QKV/SDPA), decoupled from output -> ~2x faster
                                 #   attention; output stays 256 for pts_backbone.
        num_heads=4,             # head_dim = attn_dim/num_heads = 32 (A5, flash-ok)
        # norm_cfg default = dict(type='GN', num_groups=32) -> GroupNorm, set in
        #   DGFFuser.__init__ [module-C/bn-to-groupnorm]. NOT BN2d: BatchNorm on
        #   the sparse, low-variance LiDAR BEV amplifies the norm ~18x -> NaN.
        #   GroupNorm is batch-free (unaffected by --sync_bn torch). To force BN
        #   back: norm_cfg=dict(type='BN2d').
        # [module-C/fix-residual-stability] out_proj(zero-init) + gamma(ReZero
        # 0.05) + final ReLU are built-in (not config knobs).
    ))

# Runtime products MUST live outside the (re-extracted) source tree.
# <FILL_ME>: replace with a real absolute path on your server, e.g. /data/abl/dgf
work_dir = '/data/abl/dgf'
