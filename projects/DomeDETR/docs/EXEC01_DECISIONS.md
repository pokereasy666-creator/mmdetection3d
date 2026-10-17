# 执行文档 #1 · 决策记录（预登记）

- 登记日期：2026-10-17，在会话中经用户确认
- 效力：在第 3 步开跑前登记；登记之后，不得因任何结果修改下列取值和规则
- 依据：`EXEC01_INPUTS_FROM_LOG.md`（基于 2026-07-18 训练日志的分析）
- 服务器仓库根目录（`$REPO`）：`/212022085500129/bevfusion-concat/mmdetection3d-claude-jolly-wozniak-c4YMQ`

| 编号 | 决策 | 取值 |
| --- | --- | --- |
| D-1 | 诊断对象 | CKPT = `$REPO/work_dirs/bevfusion_lidar-cam_official6e_4xa30_amp512_accum4_seed577127641/epoch_5.pth`（lidar-cam 融合模型，6 轮官方调度中末轮下跌前的第 5 轮）<br>CONFIG = 服务器上的 `projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_4xa30-amp-accum_nus-3d.py`<br>TRAIN_WORK_DIR = 上述训练目录 |
| D-2 | 参考值 | BASE_MAP = 0.6648，BASE_NDS = 0.7060（训练日志中 epoch 5 的 val 结果）。S0d 会自动核对：权重所在 epoch 的 val 日志按 `%.4f` 格式化后必须与这两个值完全相同，否则 S0d 判 FAIL |
| D-3 | S0a 配置差异判据 | 四个核心 .py 以及 `projects/BEVFusion` 下其余源码：md5 必须严格一致<br>三份配置文件：md5 不同时，比较审计版与服务器版解析后的完整配置（两边用同一份服务器上的 `default_runtime.py`）。只有当**每一处**字段差异都落在以下路径之内，并且 **S0d 通过**时，才算可接受：<br>• 调度：`param_scheduler`、`train_cfg`<br>• 优化器：`optim_wrapper`、`auto_scale_lr`<br>• batch：`train_dataloader`<br>• 训练流水线：`train_pipeline`<br>• Swin 初始化路径：`model.img_backbone.init_cfg.checkpoint`<br>其他任何字段有差异（包括 `default_hooks`、`randomness`、顶层辅助变量）都判 FAIL，停下回传，由你裁决。只改注释、没有字段差异时可以接受，报告中会注明"0 处字段变化" |
| D-4 | R1 阈值 | ΔmAP_all ≥ **1.0 pp**（ΔmAP_all = all 组 mAP − base 组 mAP）。满足：继续按 v1.1 第 10 节的顺序判读；不满足：停止 Dome-DETR 主线 |

## 对 v1.1 的填充与修订

| v1.1 位置 | 原内容 | 现在 |
| --- | --- | --- |
| 第 1 节 部署位置 | `mmdetection3d-claude-jolly-wozniak-c4YMQ` | `$REPO`（多一层 `bevfusion-concat`） |
| 第 1 节 CKPT | `baseline1/epoch_19.pth` 的绝对路径（待填） | D-1。按用户说明，`baseline1` 是 LiDAR-only，与融合 CONFIG 错配，因此不采用 |
| 第 1 节 BASE | 待填 | D-2 |
| 第 1 节 R1 | 待填 | D-4：1.0 pp |
| 第 5 节 S0a | `git diff` 为空，或 md5 逐一相同 | 源码严格一致；配置文件按 D-3 |
| 第 12 节 V6 | 0.665 / 0.696，低约 2 pp | 0.6648 / 0.7060，比论文 val（68.5 / 71.4）低 2.0 pp mAP、0.8 pp NDS |

## 运行时附带的设置（按 C3 需在回传中说明）

`CFG_OPTS="model.img_backbone.init_cfg.checkpoint=$REPO/checkpoints/swint-nuimages-pretrained.pth"`

这条设置把 Swin 初始化文件钉到训练时实际加载过的那一份。完整权重加载后会覆盖它，不影响推理；它只是让模型构建不依赖配置文件里写的是哪个路径。所有取值都集中在 `tools/query_diag/exec01_env.sh`，在仓库根目录 `source` 即可。

## 未决事项（不影响本诊断）

- `work_dirs/baseline1/epoch_19.pth` 到底是 LiDAR-only 还是融合模型。RECON 和 M0_PLAN 把它锁定为 EP-Fusion 的教师权重，需要用 `EXEC01_INPUTS_FROM_LOG.md` 第 3 节的命令核实。
