# +DGF ablation, V2 = paper-exact structure (DepthFusion Eq. 1-4).
#
# Inherits the ENTIRE fair 6-epoch recipe from the V1 DGF config (official
# 6e schedule, seed 577127641, batch2/accum4/AMP512, load_from + Swin paths)
# and swaps ONLY the fusion module: DGFFuserV1 (128-dim bottleneck + outer
# residual) -> DGFFuserV2 (native 256-wide fusion, single 80->256 image
# adapter, no Q/K/V projections, LayerNorm, output = Eq. 4's F directly).
#
# Known risk (accepted): V2 runs attention/FFN at 256 channels. An earlier
# experiment reported OOM at 256; if this run OOMs, first try
#   --cfg-options model.fusion_layer.use_checkpoint=True
# (recompute-in-backward, same math), and only then roll back to the V1
# config (plan B).
_base_ = [
    './bevfusion_lidar-cam_4xa30_dgf_v1_faithful_nus-3d.py'
]

model = dict(
    fusion_layer=dict(
        _delete_=True,
        type='DGFFuserV2',
        in_channels=[80, 256],
        num_heads=8,
        ffn_channels=256,
        use_checkpoint=False,
    ))

work_dir = 'work_dirs/dgf_v2_paperexact_6e'
