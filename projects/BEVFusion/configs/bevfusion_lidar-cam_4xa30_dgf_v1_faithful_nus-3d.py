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

# Training recipe (optimizer, lr, loss_scale, grad accumulation, clip_grad)
# is inherited UNCHANGED from the 4xA30 baseline config so that the +DGF
# ablation differs from the baseline in the fusion_layer ONLY. Do not add an
# optim_wrapper override here.

work_dir = 'work_dirs/dgf_v1_faithful'
