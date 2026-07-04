_base_ = [
    './bevfusion_lidar-cam_voxel0075_4xa30-amp-accum_nus-3d.py'
]

model = dict(
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

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.01),
    accumulative_counts=4,
    clip_grad=dict(max_norm=10, norm_type=2),
    loss_scale=64.0)

work_dir = 'work_dirs/dgf_v1_faithful'
