# EXPERIMENTS — BEVFusion LiDAR-Camera 基线复现（4×A30 适配）

> 本文件是**复现 Runbook + 实验记录**。所有「结果数字」字段在本仓库中标为 `PENDING`，
> 须在 4×A30 机器上按本文命令实跑后回填——**不得编造任何数字**。
> 本次会话的执行环境为 CPU-only 容器（无 GPU / 无 nuScenes / 无法下载权重），
> 因此训练、评测、σ 测量均**未在此执行**，仅产出可直接运行的配置与命令。

---

## 0. 目的与范围

在 `projects/BEVFusion` 的 LiDAR-camera 融合基线上：
1. 用 4×NVIDIA A30(24GB) 复现官方融合阶段基线，与官方分数对齐（±1 内）；
2. 给出方差(σ)基线协议；
3. 固定一套「公平」训练设置，供后续三个开关化模块(A/C/D)的变体在**完全相同**的条件下训练。

**官方参考分**（来源 `projects/BEVFusion/README.md` 第 71 行，8 GPU × batch 4 = 32 训练）：

| 模型 | NDS | mAP |
| --- | --- | --- |
| lidar-only | 69.6 | 64.9 |
| **lidar-cam（本次复现目标）** | **71.4** | **68.6** |

> ⚠️ 这两个数字只用于**复现保真核对**（判断 4 卡复现是否落在官方 ±1 内）。
> 它们是官方 8×A100 全量权重的成绩，**绝不能**当作后续变体的对比锚点（见 §8 公平性铁律）。

---

## 1. 硬件与软件环境

- **硬件**：4 × NVIDIA A30（24GB/卡）。
- **软件**：按 `projects/BEVFusion/README.md` 与 mmdet3d 主 README 的版本矩阵安装
  （PyTorch / CUDA / MMCV / MMDetection / MMDet3D 版本须与仓库 `requirements` 一致）。

### 1.1 环境搭建步骤（建议 conda 隔离）

```bash
# 1) 基础环境（版本以仓库 README/requirements 为准）
conda create -n bevfusion python=3.8 -y
conda activate bevfusion
# 2) PyTorch + CUDA（与 A30 驱动匹配的 CUDA 版本）
#    pip install torch==... torchvision==... --index-url https://download.pytorch.org/whl/cuXXX
# 3) OpenMMLab 栈
pip install -U openmim
mim install mmengine "mmcv>=2.0.0rc4" "mmdet>=3.0.0"
# 4) 安装本仓库 mmdet3d（在仓库根目录）
pip install -v -e .
```

### 1.2 编译 BEVFusion 自定义 CUDA 算子（**关键，且与 MMCV 不同**）

`projects/BEVFusion/README.md` 明确指出：BEVFusion 的体素化算子与 MMCV 的实现**不同**，
使用官方预训练权重时必须用其自带实现。在仓库根目录执行：

```bash
python projects/BEVFusion/setup.py develop
```

这会编译两个扩展（见 `projects/BEVFusion/setup.py`）：
- `voxel_layer` → `Voxelization` / `DynamicScatter`（`bevfusion/ops/voxel/`）
- `bev_pool_ext` → `bev_pool`（`bevfusion/ops/bev_pool/`）

> 编译需要 A30 对应的 GPU 架构（setup.py 已含 sm_70/75/80/86）。本 CPU 容器无法编译，
> 故此步骤须在 4×A30 机器执行。

---

## 2. 数据（假设已就绪）

约定 nuScenes 已按 mmdet3d 流程准备好，路径 `data/nuscenes/`，含：
`nuscenes_infos_train.pkl`、`nuscenes_infos_val.pkl`、`nuscenes_dbinfos_train.pkl`
以及 `samples/` `sweeps/`（见 lidar config 第 18–28 行 `data_prefix`）。

> 本会话容器 `data/` 下**没有** nuScenes（仅 lyft/s3dis/scannet/sunrgbd），且磁盘不足以放下完整数据集。

---

## 3. 预训练权重（跳过 LiDAR 第一阶段）

不从零训 LiDAR-only，直接用官方权重初始化融合阶段：

| 用途 | 文件名 | 下载地址 |
| --- | --- | --- |
| `load_from`（LiDAR-only 检测器整体权重） | `bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth` | `https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth` |
| `model.img_backbone.init_cfg.checkpoint`（Swin-T nuImages 主干） | `swint-nuimages-pretrained.pth` | `https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/swint-nuimages-pretrained.pth` |

二者经启动命令 `--cfg-options` 注入，**不写死在 config**（与 README 第 51 行一致）。

> 本会话容器无法下载（WebFetch/网络被拦截）；须在 4×A30 机器 `wget`/`mim download` 获取。

---

## 4. 4 卡适配设置（最终方案）

| 项目 | 官方 | 本复现（4×A30） | 说明 |
| --- | --- | --- | --- |
| GPU 数 | 8 | **4** | |
| batch / 卡 | 4 | **2** | 24GB 卡 + fp16 的已验证可行值 |
| 梯度累积 `accumulative_counts` | 1 | **4** | 4×2×4 = **有效 batch 32** |
| 有效 batch | 32 | **32** | 与官方一致 |
| 混合精度 | 关 | **开（`--amp`，fp16）** | 省显存；模型内 voxelize/view_transform 已各自管控 autocast |
| `optim_wrapper` 类型 | OptimWrapper | OptimWrapper（`--amp` 时自动转 AmpOptimWrapper） | |
| LR | AdamW 2e-4 | **AdamW 2e-4（不变）** | 见 §4.1 |
| LR 调度 | cyclic 6 epoch | **不变** | 因有效 batch 仍是 32 |
| SyncBN | 默认 BN | **开（`--sync_bn torch`）** | 见 §4.2 BN 缺口 |
| `auto_scale_lr` | enable=False | **enable=False（不变）** | 见 §4.1 |

适配 config：`projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_4xa30-amp-accum_nus-3d.py`
（继承官方 fusion config，仅 override `train_dataloader.batch_size=2` 与
`optim_wrapper.accumulative_counts=4`）。

### 4.1 LR 缩放决策：选「梯度累积」，不选 `auto_scale_lr`（二选一）

- **选择**：梯度累积（`accumulative_counts=4`），把有效 batch 精确拉回 **32**，
  从而 `lr=2e-4` 与 cyclic 调度**原样保留**。
- **理由**：本任务以「忠实复现」为第一目标。梯度累积复原了与官方一致的有效 batch 与优化动力学，
  最大化保真度；`auto_scale_lr`（按实际 batch=8 把 LR 线性缩到 2e-4×8/32=5e-5）会改变实际 LR 与
  调度形状，引入与官方不同的训练轨迹，仅作为**备选**（若显存逼迫只能用真实小 batch 时启用）。
- **备选启用方式**（若放弃梯度累积）：把 config 的 `auto_scale_lr.enable` 置 True 或启动加
  `--auto-scale-lr`，并去掉 `accumulative_counts`。**二者不可同时用**（会重复缩放）。

### 4.2 SyncBN 与 BN 统计缺口（已知近似，须记录）

- 启用 SyncBN（启动加 `--sync_bn torch`，`tools/train.py` 会把所有 BatchNorm 转成 torch SyncBatchNorm）。
- **缺口**：梯度累积**不**改善 BN 统计——每次前向每卡仍只看 batch 2，SyncBN 跨 4 卡聚合为 **8**，
  仍 < 官方的 **32**（官方每卡 4 × 8 卡）。这是 4 卡复现与官方之间**唯一无法靠累积消除**的差异，
  可能带来 <1 量级的指标抖动。若复现略低于官方且其它都对齐，应优先怀疑此项。

---

## 5. 启动命令（融合阶段训练）

设：
```bash
LIDAR_CKPT=/path/to/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth
SWINT_CKPT=/path/to/swint-nuimages-pretrained.pth
CFG=projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_4xa30-amp-accum_nus-3d.py
```

**4 卡训练**（仓库根目录）：
```bash
bash tools/dist_train.sh ${CFG} 4 \
  --amp \
  --sync_bn torch \
  --cfg-options \
    load_from=${LIDAR_CKPT} \
    model.img_backbone.init_cfg.checkpoint=${SWINT_CKPT}
```

**OOM 兜底顺序**（保持有效 batch=32 不变以保可比）：
1. `batch_size=2 → 1`，同时 `accumulative_counts=4 → 8`（4×1×8=32）；
2. 仍 OOM：开 activation checkpointing（`model.img_backbone.with_cp=True`，见 config 第 29 行该字段）；
3. 显存有余：可试 batch 3/4 并相应降 `accumulative_counts`（保持 ×4卡=32；注意 32/(4×3) 非整数，
   batch 3 时改用有效 batch 仍需可整除，优先 batch 2 或 4）。

---

## 6. 评测命令

```bash
bash tools/dist_test.sh ${CFG} /path/to/trained_fusion.pth 4
```
评测器 `NuScenesMetric`（`mmdet3d/evaluation/metrics/nuscenes_metric.py`）报告：
**mAP、NDS**、mATE、mASE、mAOE、mAVE、mAAE 及逐类 AP。

---

## 7. 基线复现结果（待回填，禁止编造）

训练融合阶段 6 epoch → 在 val 评测 → 填下表 → 与官方对比。

| 指标 | 官方(8×4=32) | 本复现(4×2×accum4=32) | Δ(复现−官方) | 状态 |
| --- | --- | --- | --- | --- |
| NDS | 71.4 | `PENDING` | `PENDING` | ⬜ |
| mAP | 68.6 | `PENDING` | `PENDING` | ⬜ |

**保真判据（铁律）**：若 |Δ| > 1（NDS 或 mAP 任一），**标红 RED 排查，不得往下做模块**。
常见排查方向：自定义算子是否编译/生效、两份预训练权重是否正确加载（看日志 `load_from` 与
missing/unexpected keys）、SyncBN 是否真的开启、AMP 是否引入数值问题、数据 infos 版本。

> 回填示例（仅格式说明，非真实数字）：`NDS 71.x / mAP 68.x，Δ=-0.x，状态 ✅`。

---

## 8. 公平性铁律

- baseline 与**所有**后续变体（+A / +C / +A+C / +A+C+D）必须在**同一套** 4×A30 / batch2 /
  amp / accumulative_counts4 / SyncBN 设置下训练。
- **绝不**拿官方 8×A100 全量权重(71.4/68.6) 当 baseline 去比 4 卡训练的变体。
- 变体的对比锚点 = **本文 §7 实跑出的 4 卡 baseline 数**（回填后）。

---

## 9. 方差(σ)基线协议（待回填）

受算力约束，按优先级：

1. **首选**：先跑 1 次忠实全量复现（§7）。
2. **有余力**：再补 2 个不同 random seed 的全量训练，得到 3 个点 → 求 NDS/mAP 的样本标准差 σ。
3. **退路（近似）**：若全量 3 seed 不现实，用**缩短 schedule**（如 `train_cfg.max_epochs=2`）
   跑 3 个 seed，估**相对** σ，并在记录中**明确标注「近似 σ（短 schedule）」**。
4. Phase-1 的「2σ 判据」至少需要一个 σ 粗估值。

| 方案 | seeds | epochs | NDS 列表 | mAP 列表 | σ(NDS) | σ(mAP) | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 全量 | `PENDING` | 6 | `PENDING` | `PENDING` | `PENDING` | `PENDING` | 首选 |
| 短 schedule 近似 | `PENDING` | 2 | `PENDING` | `PENDING` | `PENDING` | `PENDING` | 仅估相对 σ |

设置随机种子：启动加 `--cfg-options randomness.seed=<N>`（mmengine 标准）。

---

## 10. 本会话执行环境限制（透明声明）

| 能力 | 状态 | 影响 |
| --- | --- | --- |
| GPU / nvidia-smi | ❌ 无 | 无法训练/评测/编译 CUDA 算子 |
| torch / mmcv / mmdet3d | ❌ 未装 | 无法跑 config 解析（已尽量结构镜像官方降低风险） |
| nuScenes 数据 | ❌ 无（且盘不足） | 无法跑数据相关流程 |
| 网络下载（权重/论文全文） | ❌ WebFetch 403 | 无法下权重；论文公式取材自用户粘贴 |

因此 §7 / §9 的数字字段一律 `PENDING`，须在 4×A30 机器实跑回填。
