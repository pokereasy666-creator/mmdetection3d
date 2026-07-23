# [module-A/depth-sup-v2] +A v2 variant: BEVDepth-style explicit depth
# supervision WITH input-dropout to close the input-reconstruction shortcut.
#   BEVDepth (arXiv:2206.10092) depth supervision + depth-completion training.
#
# Why v2 (evidence from the paired 6-epoch ablation, same seed 577127641):
#   baseline        NDS 0.7060 / mAP 0.6648  (best ep5)
#   +A w1.0         NDS 0.7034 / mAP 0.6600  (best ep5)   ΔNDS -0.26 / ΔmAP -0.48
#   +A w3.0         NDS 0.6985 / mAP 0.6508  (best ep5)   ΔNDS -0.75 / ΔmAP -1.40
# The dose-response is monotonic but NEVER positive: tuning the weight only
# reduces the harm, it cannot make the supervision helpful. Root cause is target
# leakage -- the DepthLSSTransform feeds the same sparse LiDAR depth as BOTH the
# depthnet input AND (under v1) the supervision target, and only observed pixels
# are supervised, so the net can learn to COPY the input instead of inferring
# depth. v2 drops (1 - keep_ratio) of the INPUT points during training while the
# supervision GT keeps the FULL projection, forcing depth *completion*.
#
# Inference is UNCHANGED (dropout is train-only) and NO parameters are added, so
# the model state_dict is still identical to the baseline's; the only train-time
# differences vs baseline are the `loss_depth` term and the train-time input
# dropout. Trained under the exact same 4xA30 recipe as baseline (fairness).
#
# Offline-deployment notes (see RUNBOOK_A.md):
#   - work_dir is set OUTSIDE the source tree (placeholder below). REPLACE it.
#   - pretrained weights are injected at launch via --cfg-options with LOCAL
#     absolute paths (no online URLs).

_base_ = ['./bevfusion_lidar-cam_voxel0075_4xa30-amp-accum_nus-3d.py']

# Enable depth supervision + input dropout on the (inherited) DepthLSSTransform.
# Merges into the inherited view_transform dict; all other fields
# (dbound=[1.0,60.0,0.5] -> D=118 bins, feature_size=[32,88], etc.) are kept.
model = dict(
    view_transform=dict(
        use_depth_sup=True,
        # Weight held at the official BEVDepth 3.0 so v2 is comparable to +A
        # w3.0 (the only change vs that run is the input dropout). If depth
        # still dominates, sweep 1.0 as with v1.
        depth_loss_weight=3.0,
        # keep 30% of the input LiDAR points during training (drop 70%); the
        # supervision GT stays the FULL projection. 1.0 == v1 (no dropout).
        depth_input_keep_ratio=0.3,
    ))

# Runtime products MUST live outside the (re-extracted) source tree.
# <FILL_ME>: replace with a real absolute path, e.g. /data/abl/depthsup_v2
work_dir = '/data/abl/depthsup_v2'
