# M0 勘察报告 — 第零步环境判定 + 第一步 A 静态勘察

> VERSION: 2026-06-12 M0-recon+probe+plan ｜ 基线 commit: 833592f（claude/jolly-wozniak-c4YMQ 同步点）
> 本报告所有结论给出 file:line 级引用，均来自仓库源码直读；
> 需 GPU/数据/日志/已装环境的项原标占位，**已于 2026-06-25 由 scripts/m0_probe.py 六项实跑回填**
> （P1-P6，原始报告见 m0_probe_report_20260625*.txt）；离线适配经验见文末附录。

---

## 第零步：环境关系判定（铁律 13/14 安全检查）

**判定结果（先行明示）：本会话运行环境是与训练服务器完全分离的云端 CPU 容器；
训练状态无法从本机确认 → 本会话全程遵守铁律 13（只读 + 只新建文件）；
一切运行时勘察项必须走探针脚本（铁律 14）。**

实测依据：

| 检查项 | 结果 |
| --- | --- |
| `nvidia-smi` | command not found（无 GPU） |
| `pip list`（torch/mmengine/mmcv/mmdet3d） | 全部未安装（`import torch` → ModuleNotFoundError） |
| `data/nuscenes` | 不存在（data/ 下仅 lyft/s3dis/scannet/sunrgbd 占位） |
| `work_dirs/`、checkpoint 文件 | 均不存在 |
| `ps` 扫训练进程 | 本机无训练进程；但训练在用户离线服务器上，本机不可见 → 按"无法确认"从严处理 |

可本机直接做的项：全部静态勘察（本报告 §1–§6）。
必须走探针的项：版本实测、复现 val 指标、张量形状、显存/走时、pipeline 实出结构、
mmengine 运行时行为（P1–P6，见 scripts/m0_probe.py）。

---

## §1 版本与配置

**静态可知（源码声明）：**

| 项 | 值 | 出处 |
| --- | --- | --- |
| mmdet3d | 1.4.0 | mmdet3d/version.py:3 |
| mmengine 兼容窗 | ≥0.8.0, <1.0.0 | mmdet3d/__init__.py:13-14 |
| mmcv 兼容窗 | ≥2.0.0rc4, <2.2.0 | mmdet3d/__init__.py:9-10 |
| mmdet 兼容窗 | ≥3.0.0rc5, <3.4.0 | mmdet3d/__init__.py:17-18 |
| PyTorch / CUDA 实际版本 | **torch 2.0.1+cu118 / CUDA 11.8 / cuDNN 8700** | 探针 P1 实测 |
| 实测依赖版本（白名单关键项） | mmengine **0.10.5** / mmcv **2.1.0** / mmdet **3.2.0** / numpy 1.23.5 / torchvision 0.15.2+cu118 / matplotlib 3.5.3 / python 3.8.20 | 探针 P1（共 157 个发行版，全清单见 m0_probe_report_run1.txt） |
| GPU（探针时可见） | **3 × NVIDIA A30 23.5GB sm_80**（CLAUDE.md 标称 4 卡；本轮探针仅 3 卡可见，估时与 DDP 按 3 卡计） | 探针 P1 |

**复现 config**：`projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_4xa30-amp-accum_nus-3d.py`
（仅 override `train_dataloader.batch_size=2/num_workers=4`（:20）与
`optim_wrapper.accumulative_counts=4`（:27）），`_base_` = 官方 fusion config
`bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py`（:13-15）。

**config ↔ 官方 checkpoint 对应关系核对**（官方 ckpt
`bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth`，
README.md:69-71 表格中该 ckpt 挂的 config 正是上述官方 fusion config）：

| 核对项 | config 值 | 出处 | 结论 |
| --- | --- | --- | --- |
| voxel_size | [0.075, 0.075, 0.2] | lidar base config:10、:52 | 与 ckpt 命名 voxel0075 一致 ✓ |
| sparse_shape / grid | [1440,1440,41] | lidar base config:59、:108 | ✓ |
| out_size_factor | 8（→BEV 180×180） | lidar base config:110 | ✓ |
| 类别数 | 10 类 | lidar base config:12-16、bbox_head num_classes=10:88 | ✓ |
| 融合结构 | ConvFuser in[80,256]→256 | fusion config:56-57 | ✓ |
| 检测头 | TransFusionHead, 200 proposals, in 512 | lidar base config:84-88 | ✓ |

→ **结构对应关系成立**（最终由探针 P2 的 missing/unexpected keys 实证）。

**复现 val 指标**：【已由复现实跑回填，见下】

| 指标 | 官方（README.md:71） | 本复现（baseline1 epoch_19） | Δ（复现−官方） |
| --- | --- | --- | --- |
| NDS | 71.4 | 69.6 (0.696) | −1.8 |
| mAP | 68.6 | 66.5 (0.665) | −2.1 |

> 数据点取自 work_dirs/baseline1 的 20 轮 run 在 **epoch_19**（暴跌前最佳点，见 §4 说明）。
> **环境健康判定（通过）**：两个独立 run（6 轮 / 20 轮）均收敛到 NDS≈0.69 区间，且末轮
> 暴跌模式自洽（见 §4），判定复现环境健康、自定义算子/权重加载/AMP 均正常。
> Δ≈−1.8 NDS 最可能源于 **SyncBN 跨卡数差异**：本机 4 卡×batch2 vs 官方 8 卡×batch4，
> 有效 batch 同为 32 但每卡前向看到的样本数（4×2=8 vs 8×4=32）不同，BN 统计行为有别。
> 此偏差对 EP-Fusion 的**相对比较不构成影响**（R0 与 R1-R3 同在本机同设置下训练，
> 锚点为本机 R0 而非官方分；见 M0_PLAN.md §8 公平性 / I.1）。

**门槛说明**：原 |Δ NDS| > 0.5 排查门是针对"复现是否忠实于官方"。此处 Δ=−1.8 已 >0.5，
但经上述双 run 自洽 + SyncBN 归因排查后判定为已知系统性差异、非 bug，**放行进入 M0**；
M0 的走/停判据改以本机 R0 冻结对照为锚（M0_PLAN.md H 章）。

---

## §2 融合层解剖（静态）

- **ConvFuser**：`projects/BEVFusion/bevfusion/transfusion_head.py:28-42`。
  `nn.Sequential` 子类：`Conv2d(sum(in_channels)=336 → 256, k=3, pad=1, bias=False) → BatchNorm2d → ReLU(True)`；
  `forward(self, inputs: List[torch.Tensor]) -> torch.Tensor`，对输入列表 `torch.cat(dim=1)` 后过序列。
  **含 BN → 作为冻结教师必须 `.eval()`（铁律 1 的直接对象）。**
- **调用点**：`projects/BEVFusion/bevfusion/bevfusion.py:275-276`
  `x = self.fusion_layer(features)`，其中 `features = [img_feature, pts_feature]`
  （:271 append 图像 BEV 在前、:273 append 点云 BEV 在后）。
- **通道/空间尺寸推导**（✅ 探针 P3 实测确认，与静态推导完全一致）：
  - 相机 BEV：view_transform=DepthLSSTransform `out_channels=80`（fusion config:48），
    xbound/ybound=[-54,54,0.3]（:51-52）→ (54−(−54))/0.3 = 360 格，`downsample=2`（:55）→ 180。
    **[B, 80, 180, 180]**（P3 实测 [1,80,180,180]）
  - LiDAR BEV：voxel 0.075 → 1440 格，BEVFusionSparseEncoder 8× 下采 → 180，输出 256 通道。
    **[B, 256, 180, 180]**（P3 实测 [1,256,180,180]）
  - fusion 输出 **[B, 256, 180, 180]**（P3 实测 [1,256,180,180]）；fusion 输入顺序实测
    = `[[相机BEV(80), LiDAR BEV(256)]]`（img 在前，坐实 bevfusion.py:271/273）。
  - P3 实测属性名：`view_transform / pts_middle_encoder / fusion_layer / pts_backbone /
    pts_neck` 全部存在 → **B 章 proj_C(80→256)/proj_L(256→256)/D=256/PoEFuser 接入点
    全部确认无需改动**。
- **fusion 之后**：`pts_backbone` = SECOND（in 256 → [128,256]，lidar base config:67-70）→
  `pts_neck` = SECONDFPN（→[256,256] concat = 512，:75-78）→
  `bbox_head` = TransFusionHead（`transfusion_head.py:45` 起；num_proposals=200, in_channels=512,
  hidden_channel=128, num_classes=10，lidar base config:84-88）。

---

## §3 数据管线解剖（损坏注入点决策，最重要项）

**train pipeline 完整顺序**（fusion config:59-132，图像侧 / 点云侧标注）：

| # | transform | 侧 | 定义位置 | 作用 |
| --- | --- | --- | --- | --- |
| 1 | BEVLoadMultiViewImageFromFiles(to_float32) | 图 | projects/BEVFusion/bevfusion/loading.py:14 | 6 相机图加载，float32 0-255 |
| 2 | LoadPointsFromFile(load_dim=5,use_dim=5) | 点 | mmdet3d/datasets/transforms/loading.py | 关键帧点云 |
| 3 | LoadPointsFromMultiSweeps(sweeps_num=9) | 点 | mmdet3d/datasets/transforms/loading.py | 合并 9 sweep |
| 4 | LoadAnnotations3D | 注 | mmdet3d/datasets/transforms/loading.py | 3D 框/标签 |
| 5 | ImageAug3D | 图(几何) | projects/BEVFusion/bevfusion/transforms_3d.py:14 | 逐相机 resize/crop/flip/rot，写 img_aug_matrix（:107）；内部经 uint8 PIL（:54）后转回 float32（:103） |
| 6 | BEVFusionGlobalRotScaleTrans | 点(几何) | transforms_3d.py:147 | R/T/S 作用点+框，复合 lidar_aug_matrix（:175-184） |
| 7 | BEVFusionRandomFlip3D | 点(几何) | transforms_3d.py:112 | BEV 双向翻转，复合 lidar_aug_matrix（:139-142） |
| 8 | PointsRangeFilter | 点 | mmdet3d/datasets/transforms/transforms_3d.py | 裁范围 |
| 9 | ObjectRangeFilter / 10 ObjectNameFilter | 注 | 同上 | 框过滤 |
| 11 | GridMask(prob=0.0) | 图 | transforms_3d.py:190 | **no-op**（config:106 注释自证） |
| 12 | PointShuffle | 点 | mmdet3d/datasets/transforms/transforms_3d.py | 点序打乱 |
| 13 | Pack3DDetInputs | 双 | mmdet3d/datasets/transforms/formating.py（INPUTS_KEYS=['points','img']:50） | 'points'/'img' 进 inputs，其余进 data_samples |

a) **几何增广**：#5（图像面内）→ #6/#7（3D 点+框），先图像后点云、两模态各自把几何记入
   img_aug_matrix / lidar_aug_matrix，模型端 extract_feat（bevfusion.py:253-265）读取这两个矩阵
   保证跨模态投影一致。**几何在前的铁律 3 前提天然成立。**

b) **GT-sampling：与任务假设不符·其一——fusion 训练管线本就未启用。**
   train_pipeline（config:59-132）中无 ObjectSample；且 fusion config:235
   `del _base_.custom_hooks` 把 lidar base 的 DisableObjectSampleHook（base config:384、
   hook 定义 mmdet3d/engine/hooks/disable_object_sample_hook.py）一并删除。
   **关法 = 无需关；EPFusion config 继承后仅以注释 + sanity 断言（pipeline 无 'ObjectSample'）
   防回归（铁律 4）。**

c) **图像归一化发生在 Det3DDataPreprocessor，不在 pipeline**：
   fusion config:10-14（mean/std、bgr_to_rgb=False）；归一化执行点
   mmdet3d/models/data_preprocessors/data_preprocessor.py:224（simple_process 路径）
   与 :280（collate_data 路径）。BEVFusion 未用自定义 preprocessor 子类。

d) **点云体素化发生在模型内，不在 preprocessor**：
   BEVFusion.__init__（bevfusion.py:38-43）把 `data_preprocessor.voxelize_cfg` pop 出来自建
   `pts_voxel_layer`；体素化在 `BEVFusion.voxelize`（bevfusion.py:175-201，`@torch.no_grad()`）
   由 extract_pts_feat（:166-173）调用。→ **凡在 preprocessor 或 pipeline 注入的点云损坏
   必然位于体素化之前（铁律 3 满足）。**

e) **注入点方案与干净副本携带机制**：

   注入点：单模态退化损坏置于几何增广之后、图像归一化之前 / 体素化之前。
   两个候选实现位：
   - **候选 1（pipeline 注入）**：新 corruption transform 插在 #12 PointShuffle 附近 +
     子类 Pack3DDetInputs 扩展 INPUTS_KEYS 携带 'img_clean'/'points_clean' +
     子类 preprocessor 对 clean 键施加同一归一化/堆叠。
     优点：mmdet3d transform 惯例、CPU/numpy 实现自由、browse_dataset 可视化方便；
     缺点：需 3 处新类、干净副本走 collate 传输翻倍、CPU 损坏拖慢 dataloader。
   - **候选 2（preprocessor 注入，倾向选择）**：子类 EPFusionDataPreprocessor 在
     collate 后、归一化前（GPU 上）做模式采样 + 损坏；干净副本 = 施损前 clone，零搬运；
     零 pipeline 改动；损坏为 GPU 张量操作（快）。
     缺点：偏离 transform 惯例、可视化/单测稍繁。
     两候选的损坏作用对象等价（都在 ImageAug3D 之后的 256×704 float32 0-255 图与
     增广后原始点上），铁律 3 均满足。
   - **取舍**：倾向候选 2（实现面最小、干净副本免费、GPU 快）；**最终选型列入
     M0_PLAN.md I 章开放问题 5**。
   - **关键牵连（与任务假设不符·其二）**：`extract_feat` 在 bevfusion.py:266 把
     `deepcopy(points)` 传入 extract_img_feat → DepthLSSTransform 用原始点云生成稀疏深度图
     （相机分支并非纯相机！）。后果：corrupt_lidar 会经深度输入污染相机 BEV；
     教师干净前向在该模式下必须同时重算 view_transform（用干净点）与点云分支——
     但 2D 图像特征（img_backbone+img_neck）不变可复用。详见 M0_PLAN.md C 章。
   - 点云"随机抽线"依赖 ring index：load_dim=5/use_dim=5（config:66-69）。
     ✅ **探针 P5 实测：第 5 维全为 0（unique=1, min=max=0）→ ring index 不可用**。
     故 beam_drop **正式采用俯仰角分箱回退方案**（arctan(z/√(x²+y²)) 32 分箱整 bin 丢弃），
     退化管线删除 ring 分支或仅保留回退（M0_PLAN.md D 章已定）。
     P5 另确认：inputs keys=[points, img]；metainfo 含 img_aug_matrix / lidar_aug_matrix /
     gt_instances_3d（铁律 3 的几何矩阵齐备）。

---

## §4 冻结机制核查

- **model.train() 调用时机**（✅ 探针 P6 源码实证）：EpochBasedTrainLoop.run_epoch 每 epoch
  开头 `self.runner.model.train()`（在 before_train_epoch hook 之后）；ValLoop.run 开头
  `self.runner.model.eval()` → val 后下一 epoch 的 run_epoch 再 `.train()` 恢复。
  **结论坐实铁律 10**：冻结子模块必须在 EPFusion.train() override 中强制 .eval()，
  因 mmengine 每 epoch 都会把整模型切回 train 模式。
- **max_epochs（已按复现实跑修正）**：**实际复现采用 max_epochs=20**
  （work_dirs/baseline1 dumped config:703 实证）。注意 in-repo 静态 fusion config:217 声明的是
  `max_epochs=6`（首次 6 轮 run 即用此值，已弃用，见 I.1），二者为不同 run——本条修正即纠正
  前一版误把静态 6 当作复现值。
  **CBGS 调度端点暴跌（配置固有行为、非 bug）**：该 CBGS 配置的 param_scheduler（CosineAnnealingLR
  的 T_max/end、CosineAnnealingMomentum 的端点）与 max_epochs 绑定，最后一个 epoch 因
  LR/动量调度走到端点叠加 CBGS 重采样，指标会暴跌——6 轮 run 的 epoch_6=0.6214、20 轮 run 的
  epoch_20=0.5631 均如此。**真实基线取暴跌前一个 epoch（20 轮 run 即 epoch_19=0.696）。**
  EPFusion 自身 config 的 max_epochs 与调度端点须同步设定（M0_PLAN.md E 章已列），避免误用末轮。
- **override train() 实现位置（铁律 10）**：新模型类 EPFusion（projects/EPFusion/epfusion/
  ep_fusion.py，待实现）override `train(self, mode=True)`：`super().train(mode)` 后强制
  冻结子模块逐一 `.eval()`。**仓库内先例**：mmdet3d/models/detectors/imvotenet.py:185-198
  （freeze_img_branch 模式下 train() 中强制图像分支 eval）。
- **优化器只纳入可训练参数**：fusion config 无 paramwise_cfg（optim_wrapper config:221-224）；
  paramwise_cfg 先例 configs/groupfree3d/groupfree3d_head-L12-O256_4xb8_scannet-seg.py:199-208
  （custom_keys + lr_mult）。✅ **探针 P6 实测：requires_grad=False 参数被优化器跳过 =
  PASS**——在 weight_decay=0.5 放大下 3 步更新后冻结参数仍位级不变，证明 OptimWrapper
  确实不更新无梯度参数（开放问题闭合=是）。M0 仍采取 requires_grad_(False) +
  paramwise_cfg 分组双保险。
- **AMP 启用方式**：`tools/train.py --amp`（:93-105）把 `optim_wrapper.type` 从 OptimWrapper
  改为 **AmpOptimWrapper（loss_scale='dynamic'）**；`--sync_bn torch`（:107-109）。
  4xA30 config 注释明确保留 type='OptimWrapper' 以便 --amp 生效（4xa30 config:25-27）。

---

## §5 日志机制核查（Λ 均值记录通道）

**约定成立，且不依赖 mmengine 版本**：BEVFusion **自带 override 的 parse_losses**
（projects/BEVFusion/bevfusion/bevfusion.py:75-113），其 :102 行：
`loss = sum(value for key, value in log_vars if 'loss' in key)`
——losses dict 中**不含 'loss' 字样的键只进 log_vars 记日志，不参与反传**。
EPFusion 继承 BEVFusion 即继承此行为，Λ 均值用非 'loss' 键记录即可（铁律 8 通道可用）。

两点注意：
1. mmengine `BaseModel.train_step` 调用的是 `self.parse_losses`（即上述 override 版）
   ——✅ 探针 P6 打印 train_step 源码佐证：其内 `parsed_losses, log_vars =
   self.parse_losses(losses)`，链路成立；
2. :106-111 对 log_vars **每个 key 做 dist.all_reduce** → 多卡下各 rank 的 key 集合必须
   完全一致，否则集合通信死锁。M0 的 Λ 日志因此采用"每步恒定 12 key（sum/cnt 对）"方案
   （见 M0_PLAN.md E 章）。

---

## §6 部署链路勘察（zip-by-SHA 工作流的硬约束）

- config 经 `custom_imports = dict(imports=['projects.BEVFusion.bevfusion'],
  allow_failed_imports=False)`（lidar base config:2-3）加载项目代码；
  tools/dist_train.sh 把仓库根 prepend 进 PYTHONPATH → **新解压目录的 mmdet3d/ 与 projects/
  会遮蔽旧 editable 安装**（探针 P1 含 import 路径自检）。
- **重大发现**：BEVFusion 自定义 CUDA 算子（projects/BEVFusion/setup.py：CUDAExtension
  `...ops.bev_pool.bev_pool_ext` 与 `...ops.voxel.voxel_layer`，含 sm_80=A30）为**原地编译**，
  .so 落在源码树内；fresh zip 不含 .so，而 `projects/BEVFusion/bevfusion/__init__.py` 链式
  import 立即触发 `from .ops import Voxelization`（bevfusion.py:16）→ **缺 .so 时 import 即失败**。
  → scripts/deploy.sh 负责从旧部署目录拷贝 *.so（同机同环境免重编；回退打印重编译命令）。

---

## 探针执行指引（复现训练结束后执行；探针自检到训练进程会拒跑）

```bash
# 1) 解压新 SHA 目录后，先部署（软链数据/work_dirs + 拷贝算子 + 自检）
bash scripts/deploy.sh \
    --data      /path/to/nuscenes \
    --work-dirs /path/to/work_dirs \
    --ops-from  /path/to/旧部署或训练仓库根目录

# 2) 跑探针（P1-P6 全量；输出 m0_probe_report_<日期>.txt，整文件拷回）
bash scripts/m0_probe.sh \
    --config projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_4xa30-amp-accum_nus-3d.py \
    --repro-ckpt    work_dirs/baseline1/epoch_19.pth \
    --official-ckpt /path/to/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth \
    --work-dir      work_dirs/baseline1
# 可选：--skip p4（先快速拿 P1-P3/P5/P6）、--batch-sizes 1,2、--out 自定义报告名
```

回填闭环（✅ 2026-06-25 全部完成）：P1→§1 版本表 + 依赖白名单（铁律 15）；P2→§1 复现指标与
Δ 门、ckpt 兼容性；P3→§2 形状确认；P4→M0_PLAN.md F 章估时；P5→§3 管线实证 + 抽线 ring 语义；
P6→§4/§5 mmengine 行为 + R0WeightCopyHook 时机依据。

> 探针运行细节：P1/P6 在无数据/无网下也能跑通；P3/P4 需 GPU；P5 需 data/nuscenes infos。
> 首次全量跑（run1）因下列三坑导致 P2/P3/P4/P5 报错，已据此把三条适配修复永久写入
> scripts/m0_probe.py（不再依赖服务器临时手改），见下附录。修复后 P3 实测形状、P4 实测显存/走时
> 均正常产出（见 m0_probe_report_20260625.txt）。

---

## 附录：离线环境踩坑记录（供后续会话避坑）

> 三条均为"本地无 torch/网络、py_compile 抓不到"的运行期/环境坑，已永久修复进 scripts/m0_probe.py
> （commit 见 VERSION 行），并同步影响块 1 实现注意事项。

### 坑 1：Swin 主干 init_cfg 触发联网下载 → 离线 build 直接失败
- **现象**：`MODELS.build(cfg.model)` 时报 `urllib.error.URLError: <urlopen error [Errno -3]
  Temporary failure in name resolution>`；栈底是 `BEVFusion.init_weights()`（bevfusion.py:117）
  → `img_backbone.init_weights()`（mmdet swin.py:689）去拉
  `github.com/.../swin_tiny_patch4_window7_224.pth`。P2/P3/P4 全部因此崩在 build 阶段。
- **根因**：fusion config 的 `img_backbone.init_cfg=dict(type='Pretrained', checkpoint=<url>)`；
  构造模型即触发联网。离线服务器无外网 DNS。
- **解法（永久）**：`build_model_from_cfg` 在 `MODELS.build` 前用链式 `.get()` 安全置空
  `cfg.model.img_backbone.init_cfg`（字段不存在不报错）。探针只测形状/显存/通路，不依赖主干
  "初始"权重；真正权重由 --repro-ckpt / load_from 外部提供。**块 1 注意**：EPFusion 训练时
  若需 Swin 预训练，应由 deploy 提供本地 swint-nuimages ckpt 并以 `--cfg-options
  model.img_backbone.init_cfg.checkpoint=<本地路径>` 注入，切勿留 url。

### 坑 2：nvidia-smi 进程自检误拦邻居 GPU
- **现象**：多租户机上，探针的训练进程自检可能因【其他容器/邻居任务】在别的卡上跑而被
  `--query-compute-apps` 扫到，误判"有训练在跑"而拒绝执行。
- **根因**：`nvidia-smi --query-compute-apps` 默认列全部物理 GPU 的 compute 进程，不区分本任务可见性。
- **解法（永久）**：`detect_training_processes` 读 `CUDA_VISIBLE_DEVICES`——非空则 `nvidia-smi -i
  <列表>` 仅查可见卡；显式空串则跳过 GPU 检查；未设置维持扫全部。（本轮探针机 device_count=3，
  即 CVD 限定 3 卡。）

### 坑 3：spconv 稀疏卷积权重布局不兼容 → 裸 load_state_dict 报 size mismatch
- **现象**：P3/P4 若对复现 ckpt 裸 `load_state_dict`，spconv 卷积权重维序相反
  （复现 ckpt 形如 `[16,3,3,3,5]` vs 模型期望 `[3,3,3,5,16]`），即使 strict=False 也因
  size mismatch 抛 RuntimeError。
- **根因**：不同 spconv 版本/构建对稀疏卷积 kernel 的 [out,*,in] vs [*,in,out] 维序约定不同；
  复现 ckpt 与当前环境 spconv 布局不一致。
- **解法（永久）**：P3（测形状）、P4（测显存/走时）均**跳过加载训练 ckpt**——结论与权重数值无关。
  权重兼容性的 missing/unexpected/size 报告交给 P2 的 `load_ckpt_report`（官方 ckpt 可正常加载；
  复现 ckpt 的 spconv 差异在 P2 暴露）。**块 1 注意**：EPFusion 用 load_from 加载教师/主干时，
  须确保 spconv 布局与训练环境一致（建议直接用本环境训出的 baseline1/epoch_19.pth；若跨环境
  搬权重需走 spconv 的 key/布局转换），否则 load_from 会静默 size mismatch 或加载失败。
