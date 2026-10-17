# 执行文档 #1 · 开跑输入的确定（依据 2026-07-18 训练日志）

- 日期：2026-10-17
- 依据：用户上传的 `20260718_020825.log`，以及用户说明的三个服务器目录
- 服务器仓库根目录：`/212022085500129/bevfusion-concat/mmdetection3d-claude-jolly-wozniak-c4YMQ`（下文简称 `$REPO`）
- **状态（2026-10-17）**：第 5 节第 1–3 项已确认：权重用 A（`epoch_5`），R1 定 1.0 pp，配置差异按建议规则处理。正式登记见 `EXEC01_DECISIONS.md`；第 6 节的参数已写入 `tools/query_diag/exec01_env.sh`。

## 1. 这份日志是哪次训练

| 项 | 日志内容 |
| --- | --- |
| 模型 | lidar-cam 融合（`input_modality = dict(use_camera=True, use_lidar=True)`，带 Swin-T、DepthLSS、ConvFuser） |
| work_dir | `$REPO/work_dirs/bevfusion_lidar-cam_official6e_4xa30_amp512_accum4_seed577127641` |
| 初始化 | `load_from = $REPO/work_dirs/bevfusion_lidar_voxel0075_4xa30_accum2_seed577127641/epoch_20.pth`，一个 LiDAR-only 的 20 轮训练，**不是** `baseline1` |
| 训练设置 | 4×A30 DDP，batch 2/卡，AmpOptimWrapper（loss_scale=512），accumulative_counts=4，clip 10，6 轮官方余弦调度，`sync_bn='torch'`，seed 577127641 |
| 测试设置 | `test_cfg = val_cfg = dict()`（fp32）；val 与 test 的 dataloader、evaluator 相同；bbox_head 的 `test_cfg` 与仓库配置一致 |
| Swin 初始化路径 | `$REPO/checkpoints/swint-nuimages-pretrained.pth`（日志显示加载成功） |

各轮 val（日志打印的是 4 位小数）：

| epoch | mAP | NDS |
| --- | --- | --- |
| 1 | 0.6387 | 0.6885 |
| 2 | 0.6483 | 0.6958 |
| 3 | 0.6576 | 0.7014 |
| 4 | 0.6593 | 0.7017 |
| **5** | **0.6648** | **0.7060** |
| 6 | 0.6538 | 0.6960 |

第 6 轮（最后一轮）比第 5 轮低 1.1 pp mAP，和 RECON §4 记录的"末轮下跌"现象一致。项目已有的惯例是取下跌前的那一轮作为基线。

**与审计版本（`82c2b4e`）的差异**：仓库里的 4xa30 配置是 20 轮，Swin 路径不含 `bevfusion-concat`；这次训练用的是 6 轮和 `$REPO/checkpoints/...` 路径。这说明服务器上的 4xa30 配置文件很可能被改过，或者启动时用参数覆盖过。因此 **S0a 很可能会对这个配置文件报 DIFF**。现在 S0a 会直接打印与审计版本的逐行 diff，一轮就能看清改了什么（见第 5 节问题 3）。

## 2. 用哪个权重：需要你决定

v1.1 把 CKPT 预设为 `baseline1/epoch_19.pth`。这是我在第 0 步报告里按 RECON §1 的记载提出的，当时 RECON 把它记成 lidar-cam 复现。你现在说明 `baseline1` 是 **LiDAR-only**。如果确实如此，v1.1 的"融合 CONFIG + baseline1"就是错配：相机分支的权重全部缺失，推理结果没有意义。S0c 会因为 missing keys 判 FAIL，新增的组成检查也会直接报"checkpoint 没有相机分支"。所以这个预设需要改。

| 方案 | 权重 | val mAP / NDS | 优点 | 缺点 |
| --- | --- | --- | --- | --- |
| **A（建议）** | 本地融合 `official6e…/epoch_5.pth` | 0.6648 / 0.7060（日志） | 就是今后 Dome-DETR 各变体要对比的那个 4×A30 基线（EXPERIMENTS.md §8 公平性铁律），诊断结论可以直接对应到后续方法；有同口径的 val 日志可以作 BASE | 比论文低约 2.0 pp mAP / 0.8 pp NDS（V6 局限仍然成立） |
| A′ | 同一次训练的 `epoch_6.pth` | 0.6538 / 0.6960 | 按调度完整跑完的最终权重 | 正好处在末轮下跌点，不符合 RECON 惯例 |
| B | `checkpoints/` 下的官方融合权重 | README 记 0.686 / 0.714 | 最强的基线，基本消除 V6 局限 | 不是后续变体对比的基线；需要先做 spconv 布局转换（`tools/convert_spconv_weights.py`）；没有本环境的同口径 val 日志，BASE 要先跑一次 `tools/test.py` 才能得到 |
| C | `baseline1`（LiDAR-only） | 未知 | — | 不是 BEVFusion 融合模型；CONFIG 也要换成 lidar 配置，v1.1 的输入整体都要重定 |

建议用 **A**，即 `epoch_5`。选它的理由是它与今后的对比基线是同一套设置，而不是因为它数值最高。用 val 挑出最优轮会让 base 略偏高，这只会让 oracle 增益偏保守，不会让结论更乐观。B 可以以后单独补做一次 D1，作为"更强基线上结论是否成立"的敏感性检查，不进入本轮的预登记判读。

如果选 A，BASE 值为：`BASE_MAP=0.6648`，`BASE_NDS=0.7060`。S0d 会从 `vis_data/scalars.json` 打印 6 位小数并标出 epoch 5，用来核对这两个 4 位值。

## 3. `baseline1` 的记载矛盾（影响 EP-Fusion）

RECON §1 和 M0_PLAN I.1 把 `work_dirs/baseline1/epoch_19.pth` 记作 **lidar-cam** 复现（NDS 0.696 / mAP 0.665），并把它**锁定为 EP-Fusion 的教师和冻结主干权重**。如果它实际是 LiDAR-only，EP-Fusion 冻结的相机分支就没有经过训练，那条线的实验需要复核。这和本诊断无关，但问题严重，需要单独确认。下面这条命令只用 CPU，几秒就能判定：

```bash
cd $REPO && python -c "
import re, sys, torch
ck = torch.load(sys.argv[1], map_location='cpu')
sd = ck.get('state_dict', ck); meta = ck.get('meta', {}); cfg = meta.get('cfg', '')
print('modules :', sorted({k.split('.')[0] for k in sd}))
print('epoch   :', meta.get('epoch'))
print('modality:', re.findall(r'input_modality = .*', cfg)[:1])
print('work_dir:', re.findall(r'work_dir = .*', cfg)[:1])
" work_dirs/baseline1/epoch_19.pth
```

`modules` 里有 `img_backbone`、`view_transform`、`fusion_layer` 就是融合模型，没有就是 LiDAR-only。

## 4. R1 阈值是什么

R1 是第 3 步 oracle 实验的"总上限"门槛，用来回答一个问题：**就算 query 初始化做到完美，这个模型最多能涨多少？**

- **做法**：同一个权重跑两次完整 val。
  - base：原样推理。
  - all：测试时把每个 GT 中心所在的格子强行放进 200 个初始 query。类别仍由网络决定，分数沿用原值，不泄漏 GT 置信度。
- **ΔmAP_all** = all 的 mAP − base 的 mAP。它是"只改 query 初始化或分配"这一类方法（Dome-DETR 的 PAQI、ALIGN、FocalFormer3D 的 HIP 等）在这个权重上的**理论上限**。
- **R1 阈值**是这个上限至少要达到多少，才值得继续做 Dome-DETR。ΔmAP_all 低于阈值，就停止这条主线。因为真实方法只能拿到上限的一部分，上限太低意味着实际收益进不了论文。

**1.0 还是 0.8：我建议 1.0，不要放宽。**

1. **真实方法只能拿到上限的一部分。** 文档引用的 ALIGN 在同类 heatmap 初始化的检测头上只涨了约 0.4 mAP。
2. **重新训练的方法还要扣掉随机种子噪声。** oracle 的 base 和 all 用的是同一个权重，比较时几乎没有噪声；但真实方法要重训，必须超过种子波动才能下结论。本设置的种子标准差 σ 还没有实测（EXPERIMENTS.md §9 仍是 PENDING）。同类设置常见在 0.2–0.3 pp mAP 左右，这只是数量级参考，等实测 σ 出来后应以实测为准。
3. **按上面两点推算。** 上限 1.0 pp、方法拿到 40–50%，实际收益约 0.4–0.5 pp，大致是 2σ，刚好够写；上限放到 0.8 pp，实际收益约 0.3–0.4 pp，落在噪声范围内，期刊要求的稳健增益基本拿不到。
4. **当前基线偏弱。** 本地基线比论文低约 2 pp，弱基线上测到的初始化空间往往偏大，这本身就是不宜再放宽门槛的理由。

阈值必须在第 3 步开跑前写定，之后无论结果如何都不能改。

## 5. 需要你确认的事项

1. **权重**：A（`epoch_5`，建议）、A′、B 还是 C？
2. **R1 阈值**：1.0 pp（建议）还是 0.8 pp？
3. **S0a 的配置文件差异如何处理（可以现在预先决定，省一轮往返）。** 建议：三份配置只要逐行 diff 里的改动全部属于"训练专用字段（调度、优化器、batch、train_pipeline）或 Swin 初始化路径"，并且 **S0d 通过**，就视为可接受。理由是 S0d 已经逐项证明了 CONFIG 中所有影响评测的部分（去掉 `init_cfg` 和 `train_cfg` 后的 `model`、test 数据集与流水线、evaluator、fp16）都与权重训练时一致。四个核心 .py 源码仍然必须严格一致。如果不同意，就维持原判据：S0a 报 FAIL 即停下回传。
4. `baseline1` 的核查结果（第 3 节）。不影响本诊断，但请尽快确认。

## 6. 若选 A，具体运行参数

```bash
cd /212022085500129/bevfusion-concat/mmdetection3d-claude-jolly-wozniak-c4YMQ
export CONFIG=projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_4xa30-amp-accum_nus-3d.py
export CKPT=$PWD/work_dirs/bevfusion_lidar-cam_official6e_4xa30_amp512_accum4_seed577127641/epoch_5.pth
export TRAIN_WORK_DIR=$PWD/work_dirs/bevfusion_lidar-cam_official6e_4xa30_amp512_accum4_seed577127641
# Swin 初始化路径钉到本机确实存在的文件（日志证实可以加载）。完整权重加载后会覆盖它，不影响推理。
# 这样无论服务器上的配置文件里写的是哪个路径，都不会在构建模型时失败。按 C3 规定，CFG_OPTS 非空要在回传中说明。
export CFG_OPTS="model.img_backbone.init_cfg.checkpoint=$PWD/checkpoints/swint-nuimages-pretrained.pth"
export BASE_MAP=0.6648 BASE_NDS=0.7060
```

之后按 `EXEC01_V1.1_IMPLEMENTATION.md` 第 5 节的顺序执行（s0 → 部署 + 门禁 → quick + merge → recall + merge → oracle + summary）。

**耗时估计**（按日志推算：val 时单卡约 0.3 s/帧，nuScenes 评测约 9 分钟；单卡峰值显存约 2.7 GB）：
- S0：几分钟（CPU）
- quick：约 3–5 分钟
- recall：约 30–35 分钟
- oracle：每组约 40 分钟，5 组共约 3.3 小时
- 若触发 S5 裁决：再加约 40 分钟
