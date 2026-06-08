# [module-A/depth-sup] +A variant: BEVFusion LiDAR-camera with BEVDepth-style
# explicit depth supervision on the DepthLSSTransform depth logits.
#   BEVDepth (arXiv:2206.10092) depth supervision.
#
# Inherits the verified 4xA30 baseline config UNCHANGED (batch_size 2,
# accumulative_counts 4 -> effective batch 32, AMP, SyncBN, cyclic-6e, lr 2e-4)
# and ONLY turns on `use_depth_sup` (+ depth_loss_weight) on the view transform.
# This keeps +A trained under the exact same hardware/optimisation setup as the
# baseline (fairness rule). Toggling is a single config field; with
# use_depth_sup=False the model is byte-identical to the baseline.
#
# Module A adds NO trainable parameters (it only supervises depth logits the
# baseline depthnet already predicts), so the +A model's state_dict is
# identical to the baseline's; the only training-time difference is the extra
# `loss_depth` term.
#
# Offline-deployment notes (see RUNBOOK_A.md):
#   - work_dir is set OUTSIDE the source tree (placeholder below). REPLACE it.
#   - pretrained weights are injected at launch via --cfg-options with LOCAL
#     absolute paths (no online URLs).

_base_ = ['./bevfusion_lidar-cam_voxel0075_4xa30-amp-accum_nus-3d.py']

# Enable depth supervision on the (inherited) DepthLSSTransform. Merges into the
# inherited view_transform dict, adding two keys; all other view_transform
# fields (dbound=[1.0,60.0,0.5] -> D=118 bins, feature_size=[32,88], etc.) are
# kept as-is.
model = dict(
    view_transform=dict(
        use_depth_sup=True,
        depth_loss_weight=0.5,  # BEVDepth-style default; tune if it imbalances
    ))

# Runtime products MUST live outside the (re-extracted) source tree.
# <FILL_ME>: replace with a real absolute path on your server, e.g. /data/abl/depthsup
work_dir = '/data/abl/depthsup'
