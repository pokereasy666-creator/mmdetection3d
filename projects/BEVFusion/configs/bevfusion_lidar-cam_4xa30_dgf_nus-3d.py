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
        embed_dims=256,          # deliberate adaptation: paper uses C=128; we use
                                 #   256 to match LiDAR BEV / pts_backbone (no extra
                                 #   projection). 8 heads -> head_dim 32 (A5).
        num_heads=8,             # head_dim = 32 (A5)
        attn_resolution=None,    # FULL 180x180 attention (faithful, no downsample).
                                 #   The avg-pool/interpolate knob stays dormant for
                                 #   the later speed task; None = structural no-op.
        depth_after_qknorm=False,  # A16: order of depth-encoding D vs the query
                                 #   l2norm. False (default) = current behaviour (D
                                 #   before l2norm -> l2norm strips D's depth
                                 #   modulation). Override for the A/B with
                                 #   `--cfg-options model.fusion_layer.depth_after_qknorm=True`
                                 #   to keep D AFTER l2norm (Eq.3 near/far sharpening
                                 #   survives into the logits).
        # norm_cfg default = None -> dict(type='GN', num_groups=32) = GroupNorm
        #   (A8): a feature-map norm (stats shared across space) -> preserves the
        #   fg/bg contrast the heatmap needs. Channel-wise LayerNorm was REJECTED
        #   by the 850-step smoke (per-cell LN forces contrast=1.000, freezes the
        #   heatmap). GN has no batch stats / no cross-GPU sync. Override: BN2d.
        # Faithful aggregation: U=N(V̂+V_B), F=N(FFN(U)+U), added 1:1 (no gamma
        # gate, no zero-init out_proj, no final ReLU). BUG-2: attention is
        # scaled-cosine (q,k L2-norm per head + learnable bounded scale, A15) so
        # V̂ stays ~ V_B scale -- stability is NEVER from suppressing the camera.
    ))

# [module-C/dgf-stability] loss_scale = 64 is a STARTING point, not a fixed param.
# The inherited baseline uses 512 (tuned for ConvFuser); the faithful DGF has the
# camera fully active from step 0, so the grad-magnitude distribution differs. Rule
# from the smoke run: if inf appears -> drop to 32; if healthy -> keep 64 (later
# 64->128->...->512 is fair game once stable). accumulative_counts/clip_grad are
# inherited from the baseline (effective batch 32, grad-clip max_norm 10). This is
# a DGF-config-only override; the baseline config is untouched.
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale=64.0,
    accumulative_counts=4,
    clip_grad=dict(max_norm=10, norm_type=2))

# Runtime products MUST live outside the (re-extracted) source tree.
# <FILL_ME>: replace with a real absolute path on your server, e.g. /data/abl/dgf
work_dir = '/data/abl/dgf'
