# EP-Fusion M0 主 config（R1-R3：PoE 融合 + 教师对照 NLL + 退化演练）。
# 继承 4xA30 复现 config（batch2 + accum4 + AMP 配方原样），仅叠加 EPFusion 所需 override。
# R1-R3 通过 `--cfg-options model.w_teach=0.1|1.0|10.0` 切换，无需改文件（铁律 17）。
#
# 启动（从全新解压目录；deploy.sh 完成软链+算子后）：
#   bash tools/dist_train.sh \
#       projects/EPFusion/configs/epfusion_m0_poe_4xa30-amp-accum_nus-3d.py 3 \
#       --amp --sync_bn torch \
#       --cfg-options model.w_teach=1.0 randomness.seed=2026
_base_ = ['../../BEVFusion/configs/'
          'bevfusion_lidar-cam_voxel0075_4xa30-amp-accum_nus-3d.py']

# custom_imports 必须显式写全（mmengine dict 合并对该字段整体替换）：
custom_imports = dict(
    imports=['projects.BEVFusion.bevfusion', 'projects.EPFusion.epfusion'],
    allow_failed_imports=False)

# GT-sampling：_base_ 融合管线本就无 ObjectSample（M0_RECON_REPORT §3b），无需关，
# 仅由 sanity_lambda_logging 断言 pipeline 无 'ObjectSample' 防回归（铁律 4）。

model = dict(
    type='EPFusion',
    fusion_mode='poe',
    w_teach=1.0,            # R1=0.1 / R2=1.0 / R3=10.0，经 --cfg-options 切换
    w_fused_mse=0.0,        # 可选融合 MSE 锚定旁路，默认关（开放问题 I-7）
    poe_cfg=dict(
        in_channels_cam=80,     # 探针 P3 坐实
        in_channels_lidar=256,  # 探针 P3 坐实
        embed_dims=256,         # = pts_backbone 入口，免出口投影
        lambda_hidden=64,
        gn_groups=8,
        clamp_min=-7.0,
        clamp_max=7.0,
        eps=1e-6,
        lambda_input='raw',     # 投影前原始分支特征（开放问题 I-6 开关）
        norm='GN'),             # BN 在模式混合 batch 上高危，否决；GN
    # 离线兼容（铁律 15 / RECON 附录坑1）：load_from 提供 img_backbone 权重，
    # 置空 init_cfg 阻止 BEVFusion.init_weights() 联网下载 Swin 预训练权重。
    img_backbone=dict(init_cfg=None),
    data_preprocessor=dict(
        type='EPFusionDataPreprocessor',
        mode_probs=(0.5, 0.25, 0.25),   # clean / corrupt_cam / corrupt_lidar
        p_zero=0.05,                    # 损坏模式内抽中整路置零的概率
        severity_range=(0.1, 1.0),
        seed=2026,
        emit_clean=True))               # R1-R3 需教师干净副本

# 教师/冻结主干初值（M0_PLAN I.1 锁定；从全新解压目录的相对路径）
load_from = 'work_dirs/baseline1/epoch_19.pth'

# 轮数 3（主干冻结，仅小容量 Λ 头/投影从零学 + BEV encoder/头 0.1xLR 微调）。
# param_scheduler 端点须与 max_epochs=3 同步（漏改会越界，sanity 打印首步 LR/momentum 核对）。
train_cfg = dict(max_epochs=3)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.33333333, by_epoch=False,
         begin=0, end=500),
    dict(type='CosineAnnealingLR', begin=0, T_max=3, end=3, by_epoch=True,
         eta_min_ratio=1e-4, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', eta_min=0.85 / 0.95, begin=0, end=1.2,
         by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', eta_min=1, begin=1.2, end=3,
         by_epoch=True, convert_to_iter_based=True),
]

# 优化器分组（E 章 / P6 核验）：新模块 poe_fuser 全 LR；解冻预训练部分 0.1xLR。
# 冻结参数（requires_grad=False）被优化器自动跳过（P6 实测 PASS），无需 lr_mult=0 兜底。
optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys={
        'poe_fuser': dict(lr_mult=1.0),
        'pts_backbone': dict(lr_mult=0.1),
        'pts_neck': dict(lr_mult=0.1),
        'bbox_head': dict(lr_mult=0.1),
    }))

# batch 不动：沿用 _base_ batch=2 / accum=4。P4 "显存可设4" 未含教师额外前向开销，
# 留块1部署后首训实测再调（约束6）。
randomness = dict(seed=2026)
