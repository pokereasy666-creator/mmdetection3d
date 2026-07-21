# +DGF ablation config (channel-adapted DepthFusion Depth-GFusion).
#
# FAIRNESS ANCHOR: the reproduced BEVFusion LiDAR-camera baseline
#   work_dir: bevfusion_lidar-cam_official6e_4xa30_amp512_accum4_seed577127641
#   (official 6-epoch recipe, seed 577127641, 4xA30: batch2/GPU + accum4 + AMP
#    loss_scale=512.0). This +DGF run reproduces that baseline recipe EXACTLY
#   and differs from it in the fusion_layer ONLY (ConvFuser -> DGFFuserV1).
#
# We therefore inherit the OFFICIAL 6-epoch base directly (max_epochs=6,
# param_scheduler, GridMask max_epoch=6, pipeline, auto_scale_lr disabled are
# all inherited unchanged) and re-declare only the 4xA30 training deltas that
# the baseline log used. Do NOT inherit the 20-epoch 4xa30-amp-accum config:
# its schedule was stretched to 20 epochs and would break the comparison.
_base_ = [
    './bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
]

# --- 4xA30 training deltas: byte-for-byte identical to the baseline log ---

# Seed pinned to the baseline's seed for a reproducible, aligned comparison.
randomness = dict(seed=577127641, deterministic=False)

# per-GPU batch 2 (official upstream is 8xb4 = batch 4/GPU; 4xA30 has 24GB).
# accumulative_counts=4 (below) keeps the effective batch at 4x2x4 = 32,
# matching auto_scale_lr base_batch_size=32 (inherited, disabled).
train_dataloader = dict(batch_size=2, num_workers=4)

# AMP with a STATIC loss_scale=512.0 (avoids the dynamic-scale fp16 overflow),
# accum=4, clip_grad max_norm 35->10. lr/wd inherited-equal (AdamW 2e-4/0.01).
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale=512.0,
    accumulative_counts=4,
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.01),
    clip_grad=dict(max_norm=10, norm_type=2))

# Offline hardening + fusion-layer swap. The Swin path and load_from mirror the
# baseline log; both are overridable at launch via --cfg-options if paths move.
model = dict(
    img_backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/212022085500129/bevfusion-concat/mmdetection3d-DGF-V1/checkpoints/swint-nuimages-pretrained.pth'  # noqa: E501
        )),
    # DepthFusion DGF adapted through a 128-dim attention bottleneck; same
    # [80, 256] -> 256 channel contract as the baseline ConvFuser.
    fusion_layer=dict(
        type='DGFFuserV1',
        in_channels=[80, 256],
        embed_dims=128,
        out_channels=256,
        num_heads=8,
        norm_cfg=dict(type='GN', num_groups=32),
        ffn_channels=128,
        zero_init_out_proj=False,
    ))

# Initialize from the LiDAR-only detector (two-stage BEVFusion recipe), exactly
# as the baseline did.
load_from = '/212022085500129/bevfusion-concat/mmdetection3d-claude-jolly-wozniak-c4YMQ/work_dirs/bevfusion_lidar_voxel0075_4xa30_accum2_seed577127641/epoch_20.pth'  # noqa: E501

work_dir = 'work_dirs/dgf_v1_faithful_6e'
