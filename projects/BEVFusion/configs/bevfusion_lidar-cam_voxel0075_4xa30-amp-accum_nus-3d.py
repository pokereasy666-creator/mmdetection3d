# 4xA30 (24GB/card) reproduction config for the BEVFusion LiDAR-camera fusion
# baseline. It inherits the official 8x4 fusion config and ONLY overrides the
# hardware-adaptation knobs, so the recipe stays faithful to the original.
# See projects/BEVFusion/EXPERIMENTS.md for the full rationale + launch commands.
#
# Official reference (8 GPU x batch 4 = 32):  NDS 71.4 / mAP 68.6
#   source: projects/BEVFusion/README.md, "Results and models" table.
#
# Effective batch is preserved at 32 via gradient accumulation:
#   4 GPUs  x  batch_size 2  x  accumulative_counts 4  = 32
# so the original AdamW lr=2e-4 + cyclic schedule remain valid UNCHANGED.

_base_ = [
    './bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
]

# --- per-GPU batch: 4 -> 2 (validated-feasible value on a 24GB card in fp16) ---
# Only batch_size changes; the (CBGS-wrapped) dataset definition and num_workers
# are inherited untouched. val_dataloader stays at the inherited batch_size=1.
train_dataloader = dict(batch_size=2, num_workers=4)

# --- gradient accumulation to restore the effective batch of 32 ---
# `accumulative_counts` is a standard mmengine OptimWrapper field. The optimizer,
# lr (2e-4) and clip_grad are inherited unchanged; we only add the counter.
# NOTE: keep type='OptimWrapper' so the `--amp` CLI flag (tools/train.py) can
# convert it to 'AmpOptimWrapper' (loss_scale='dynamic') at launch time.
optim_wrapper = dict(accumulative_counts=4)

# auto_scale_lr stays DISABLED (inherited: dict(enable=False, base_batch_size=32)).
# The gradient-accumulation route already keeps the effective batch at the
# base_batch_size of 32, so the LR must NOT be auto-rescaled on top of that.
# (auto_scale_lr is documented in EXPERIMENTS.md as the rejected alternative.)

# SyncBN: enable at launch with `--sync_bn torch`; tools/train.py then converts
# every BatchNorm to torch SyncBatchNorm. With batch_size 2/card the cross-card
# BN statistics are pooled over 4x2 = 8 samples -- still below the official 32;
# see the "BN statistics gap" caveat in EXPERIMENTS.md. We deliberately do NOT
# hard-code norm_cfg here, to keep the diff vs the official config minimal.
