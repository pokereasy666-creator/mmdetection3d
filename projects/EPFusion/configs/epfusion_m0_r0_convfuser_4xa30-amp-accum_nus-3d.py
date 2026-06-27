# EP-Fusion M0 R0 对照 config（冻结主干 + 可训练 ConvFuser 副本 + 同退化数据流，无 Λ/无 L_teach）。
# 公平锚点：与 R1-R3 唯一差别 = 融合机制(ConvFuser vs PoE) + 有无 L_teach；退化数据流相同。
# student_fuser 初值由 R0WeightCopyHook 在 iter0 从冻结教师 fusion_layer 拷贝。
#
# 启动：
#   bash tools/dist_train.sh \
#       projects/EPFusion/configs/epfusion_m0_r0_convfuser_4xa30-amp-accum_nus-3d.py 3 \
#       --amp --sync_bn torch --cfg-options randomness.seed=2026
_base_ = ['./epfusion_m0_poe_4xa30-amp-accum_nus-3d.py']

model = dict(
    fusion_mode='convfuser',   # loss 走 student_fuser 分支（进计算图、可训练）
    w_teach=0.0,
    poe_cfg=None,              # R0 不建 poe_fuser
    # emit_clean=False：R0 无教师前向、不消费 clean 副本，省每步第二次 clean 图归一化（阻断2）
    data_preprocessor=dict(emit_clean=False))

# before_train 时机从 fusion_layer 拷权重给 student_fuser（仅 iter0，resume 守卫见 hooks.py）
custom_hooks = [dict(type='R0WeightCopyHook')]

# R0 优化器分组：student_fuser（自教师拷初值）低 LR 微调；poe_fuser 键不存在则无影响。
optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys={
        'student_fuser': dict(lr_mult=0.1),
        'pts_backbone': dict(lr_mult=0.1),
        'pts_neck': dict(lr_mult=0.1),
        'bbox_head': dict(lr_mult=0.1),
    }))
