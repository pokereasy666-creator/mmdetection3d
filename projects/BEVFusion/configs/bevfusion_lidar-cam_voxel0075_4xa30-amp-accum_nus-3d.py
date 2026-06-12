# 4xA30 (24GB/card) training config for the BEVFusion LiDAR-camera fusion
# baseline. Inherits the official fusion config; overrides below.
#
# [baseline/20epoch-stable] This config deviates from the official 6-epoch
# recipe in exactly three regards (everything else inherited unchanged):
#   1. schedule: 6 -> 20 epochs, with the LR/momentum schedulers and the
#      GridMask max_epoch stretched consistently (same 40%/60% momentum split:
#      2.4/6 -> 8/20);
#   2. AMP stability (fixes grad_norm nan/inf): AmpOptimWrapper is pinned IN
#      the config with loss_scale=dict(init_scale=512) -- the `--amp` CLI flag
#      would otherwise inject loss_scale='dynamic' (GradScaler starting at
#      65536, which overflows early fp16 grads); clip_grad.max_norm 35 -> 10;
#   3. offline hardening: img_backbone.init_cfg.checkpoint pinned to a LOCAL
#      absolute path placeholder (never an online URL).
#
# Official reference (8 GPU x batch 4 = 32, 6-epoch recipe): NDS 71.4/mAP 68.6
# (projects/BEVFusion/README.md). With the 20-epoch schedule that comparison
# becomes approximate; this 4xA30/20e baseline is the anchor all module
# ablations must be compared against, under these exact settings (fairness).
#
# UNCHANGED knobs (do not touch): batch_size 2/GPU, accumulative_counts 4
# (effective batch 4x2x4=32), AdamW lr=2e-4/wd=0.01, SyncBN via the
# `--sync_bn torch` CLI flag, all data augmentation except GridMask.max_epoch.

_base_ = [
    './bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
]

point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
backend_args = None

# --- offline hardening: Swin-T init from a LOCAL file, never an online URL ---
# <FILL_ME>: replace <SWINT_CKPT_PATH> with the real absolute path on the
# server, e.g.
#   /212022085500129/mmdetection3d-claude-jolly-wozniak-c4YMQ/checkpoints/swint-nuimages-pretrained.pth
# Launching with
#   --cfg-options model.img_backbone.init_cfg.checkpoint=...
# still overrides this value.
model = dict(
    img_backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='<SWINT_CKPT_PATH>')))

# Full copy of the parent train_pipeline -- mmengine replaces (does not merge)
# lists, so changing one field requires re-stating the whole list. The ONLY
# change vs the parent is GridMask max_epoch 6 -> 20 (kept in sync with
# train_cfg.max_epochs). The parent's test_pipeline has no GridMask and is
# inherited untouched.
train_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.38, 0.55],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[-5.4, 5.4],
        rand_flip=True,
        is_train=True),
    dict(
        type='BEVFusionGlobalRotScaleTrans',
        scale_ratio_range=[0.9, 1.1],
        rot_range=[-0.78539816, 0.78539816],
        translation_std=0.5),
    dict(type='BEVFusionRandomFlip3D'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    # As in the parent config: 'GridMask' is not actually used (prob=0.0);
    # max_epoch merely follows train_cfg.max_epochs (6 -> 20).
    dict(
        type='GridMask',
        use_h=True,
        use_w=True,
        max_epoch=20,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=1,
        prob=0.0,
        fixed_prob=True),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes',
            'gt_labels'
        ],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans', 'img_aug_matrix',
            'lidar_aug_matrix', 'num_pts_feats'
        ])
]

# --- per-GPU batch: 4 -> 2 (validated-feasible value on a 24GB card in fp16);
# the training pipeline is re-bound to the 20-epoch copy above. Dataset type/
# modality/CBGS wrapper etc. are inherited untouched. val_dataloader stays at
# the inherited batch_size=1.
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(dataset=dict(pipeline=train_pipeline)))

# [baseline/20epoch-stable] schedule stretched 6 -> 20 epochs (full list copy;
# lists replace, not merge). Shape preserved from the parent: 500-iter linear
# warmup, cosine LR over the whole run, cyclic momentum turning at 40% of
# training (parent 2.4/6 -> 8/20 here).
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
        T_max=20,
        end=20,
        by_epoch=True,
        eta_min_ratio=1e-4,
        convert_to_iter_based=True),
    # momentum: anneal to 0.85/0.95 during the first 8 epochs, back to 1 over
    # the remaining 12 (same 40%/60% split as the parent's 2.4/6).
    dict(
        type='CosineAnnealingMomentum',
        eta_min=0.85 / 0.95,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        eta_min=1,
        begin=8,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True)
]

train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=1)

# --- AMP + gradient accumulation + stability ---
# AmpOptimWrapper is pinned IN the config: tools/train.py's `--amp` flag only
# injects loss_scale='dynamic' when converting a plain OptimWrapper; with the
# type already AmpOptimWrapper it just warns "AMP training is already enabled"
# and leaves loss_scale alone -- which is what lets init_scale=512 stick.
# loss_scale=dict(init_scale=512): still a dynamic GradScaler, but starting at
# 512 instead of 65536, so early fp16 steps don't overflow (grad_norm nan).
# clip_grad.max_norm 35 -> 10 further suppresses gradient spikes.
# Optimizer (AdamW lr=2e-4, wd=0.01) inherited UNCHANGED;
# accumulative_counts=4 keeps the effective batch at 32 (4 GPUs x 2 x 4).
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale=dict(init_scale=512),
    accumulative_counts=4,
    clip_grad=dict(max_norm=10, norm_type=2))

# auto_scale_lr stays DISABLED (inherited: dict(enable=False, base_batch_size=32)).
# Gradient accumulation already restores the effective batch of 32, so the LR
# must NOT be auto-rescaled on top of that.

# SyncBN: enable at launch with `--sync_bn torch` (unchanged).
