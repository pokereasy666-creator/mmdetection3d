# [module-C/dgf-fp32test] TEMPORARY fp32 control for the DGF backward-NaN study.
#
# Identical to the +C config (bevfusion_lidar-cam_4xa30_dgf_nus-3d.py) EXCEPT
# the optim_wrapper is swapped from AmpOptimWrapper(loss_scale=512.0) to a plain
# fp32 OptimWrapper (no AMP, no loss_scale). Everything else -- DGFFuser,
# optimizer (AdamW lr=2e-4/wd=0.01), accumulative_counts=4, clip_grad
# (max_norm=10), schedule, data -- is inherited UNCHANGED.
#
# Purpose: separate the two possible causes of DGF's grad_norm=NaN:
#   (i)  the GroupNorm/SyncBN-dominated backward path being numerically
#        ill-posed on its own (would still NaN in fp32), vs
#   (ii) large DGF gradients + fp16 overflow at the static loss_scale=512
#        (would DISAPPEAR in fp32).
# Run WITHOUT --amp. Delete this config after the experiment.

_base_ = ['./bevfusion_lidar-cam_4xa30_dgf_nus-3d.py']

# _delete_=True drops the inherited AmpOptimWrapper dict ENTIRELY so the
# loss_scale=512.0 key cannot leak into OptimWrapper (which does not accept it
# -> TypeError). optimizer / accumulative_counts / clip_grad reproduce the +C
# values verbatim; only the AMP wrapper + static loss_scale are removed (pure
# fp32).
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.01),
    accumulative_counts=4,
    clip_grad=dict(max_norm=10, norm_type=2))

# Separate work_dir so the fp32 run does not clobber the +C run's products.
work_dir = '/data/abl/dgf_fp32test'   # <FILL_ME: absolute path OUTSIDE the source tree>
