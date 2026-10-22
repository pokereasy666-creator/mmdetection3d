# Exec doc #3 v1.1 sec. 3: the fine-tuning protocol shared by every run.
# Inherits the base config of exec docs #1-#2 (md5 8f49ac79...); only the
# items below change. N (ql_epochs) is resolved when the file is parsed and
# appears in five places; if step 0 sets N = 1, the step-1 entry overrides
# all five together via --cfg-options.
_base_ = [
    '../../BEVFusion/configs/bevfusion_lidar-cam_voxel0075_4xa30-amp-accum_nus-3d.py'  # noqa: E501
]
custom_imports = dict(
    imports=['projects.BEVFusion.bevfusion', 'projects.QueryLoc.queryloc'],
    allow_failed_imports=False)

ql_epochs = 2

# Frozen: camera branch + LiDAR branch (requires_grad False and eval mode).
model = dict(
    type='QueryLocBEVFusion',
    frozen_modules=[
        'img_backbone', 'img_neck', 'view_transform', 'pts_voxel_layer',
        'pts_voxel_encoder', 'pts_middle_encoder'
    ])

# Peak lr 2e-5 (0.1 x the 2e-4 of the 6-epoch fusion run), 500 dataloader
# iters linear warmup from 1/3, cosine to 0 over N epochs, no momentum
# schedule (AdamW beta1 stays at its default 0.9).
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.33333333,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=ql_epochs,
        end=ql_epochs,
        by_epoch=True,
        eta_min=0.0,
        convert_to_iter_based=True)
]
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale=512.0,
    accumulative_counts=4,
    optimizer=dict(type='AdamW', lr=2e-5, weight_decay=0.01),
    clip_grad=dict(max_norm=10, norm_type=2))

# Validate and save only after the last epoch.
train_cfg = dict(by_epoch=True, max_epochs=ql_epochs, val_interval=ql_epochs)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(
        type='CheckpointHook', interval=ql_epochs, max_keep_ckpts=1))

# 4 GPUs x batch 2 x accumulation 4 = 32, SyncBN (as the start weights).
sync_bn = 'torch'
randomness = dict(seed=0, deterministic=False)
