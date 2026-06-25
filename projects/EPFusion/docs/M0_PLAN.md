# EP-Fusion M0 实现计划（A–I 章）

> VERSION: 2026-06-12 M0-recon+probe+plan ｜ 静态依据见同目录 M0_RECON_REPORT.md
> 本文为唯一权威版本（用户审查修订 7+1 点与三项追加已物理并入正文）。
> 依赖探针的数字原标占位，**已于 2026-06-25 由探针 P1-P6 实跑回填**（无未决占位；详见
> M0_RECON_REPORT.md §1/§3/§4 与本文 B/D/E/F 章）。
> 约束总纲：projects/BEVFusion/CLAUDE.md 铁律 1–18 + M0 范围纪律（只做逐格标量 Λ、
> 教师对照 NLL + 退化演练、PoE 融合；不做 NIG/CI/通道组 Λ/任务误差路线/nuScenes-C 正式评测/
> 注意力融合基线/DAL）。

---

## A. 工程结构

全部新代码在 projects/EPFusion/ 与 scripts/（铁律 11：不改 projects/BEVFusion/ 与 mmdet3d/ 任何文件）：

```
<repo 根>/
├── VERSION                                   # 每次 commit 更新一行"日期+模块名"（铁律 16）
├── scripts/
│   ├── deploy.sh                             # 已交付：拷贝 ops *.so / 软链 data+work_dirs / VERSION / 自检
│   ├── m0_probe.py                           # 已交付：P1–P6 探针
│   └── m0_probe.sh                           # 已交付：PYTHONPATH 包装
└── projects/EPFusion/
    ├── epfusion/
    │   ├── __init__.py                       # 显式 import 注册：EPFusion / PoEFuser /
    │   │                                     #   EPFusionDataPreprocessor / R0WeightCopyHook
    │   ├── ep_fusion.py                      # class EPFusion(BEVFusion) @MODELS.register_module()
    │   ├── poe_fuser.py                      # PoEFuser（proj_C/proj_L + lambda_head_C/L + PoE 闭式融合）
    │   ├── corruptions.py                    # TRAIN_CORRUPTIONS 注册表（纯 torch/numpy，铁律 15）
    │   │                                     #   + TEST_CORRUPTIONS_DOC（仅注释，铁律 5）
    │   ├── preprocessor.py                   # EPFusionDataPreprocessor(Det3DDataPreprocessor)
    │   └── hooks.py                          # R0WeightCopyHook（见 F 章）
    ├── configs/
    │   ├── epfusion_m0_poe_4xa30-amp-accum_nus-3d.py        # 主 config（R1-R3 共用）
    │   └── epfusion_m0_r0_convfuser_4xa30-amp-accum_nus-3d.py  # R0 对照
    ├── scripts/
    │   ├── sanity_lambda_logging.py          # G-1
    │   ├── sanity_p1.py                      # G-2
    │   ├── sanity_p4.py                      # G-3
    │   └── eval_quarter_val.py               # G-4
    ├── docs/{M0_RECON_REPORT.md, M0_PLAN.md}
    └── EXPERIMENTS.md                        # 运行矩阵记录（实跑后回填，禁止编造）
```

config 要点：
- 主 config `_base_ = ['../../BEVFusion/configs/bevfusion_lidar-cam_voxel0075_4xa30-amp-accum_nus-3d.py']`
  （batch 2 + accum 4 + AMP 配方原样继承）。
- **custom_imports 必须显式写全**（mmengine dict 合并对该字段整体替换）：
  `custom_imports = dict(imports=['projects.BEVFusion.bevfusion', 'projects.EPFusion.epfusion'], allow_failed_imports=False)`。
- GT-sampling：fusion 管线本就未启用（M0_RECON_REPORT.md §3b）→ 新 config 仅注释说明 +
  sanity_lambda_logging 断言 pipeline 无 'ObjectSample'（铁律 4 防回归）。
- 主 config 覆盖：`model.type='EPFusion'`、`model.data_preprocessor.type='EPFusionDataPreprocessor'`
  （dict 合并保留 mean/std/voxelize_cfg）、`load_from=work_dirs/baseline1/epoch_19.pth`、`train_cfg.max_epochs`、
  `param_scheduler` 端点、`optim_wrapper.paramwise_cfg`、`randomness=dict(seed=2026)`。
  所有可调项（w_teach、p_zero、严重度区间、模式配比、开关）皆为 model/preprocessor 的 config 字段，
  经 `--cfg-options` 命令行覆盖（铁律 17，零代码迭代）。

## B. 新模块设计

### B.1 PoEFuser（poe_fuser.py；全部新参数收于此，ckpt 命名空间干净）

- **proj_C**：1×1 Conv，80 → D；**proj_L**：1×1 Conv，256 → D；共享维 **D=256**
  （= pts_backbone 输入通道，免出口投影）。✅ 探针 P3 实测确认：相机 BEV 80ch、LiDAR BEV
  256ch、fusion 入/出 256ch，**80/256/256 全部坐实，无需 out-proj 兜底**。
- **lambda_head_C / lambda_head_L**（结构相同、不共享权重）：
  `Conv3×3(C_in→64) → GN(8,64) → ReLU → Conv3×3(64→64) → GN(8,64) → ReLU → Conv1×1(64→1)`，
  输出 1 通道 **log-precision**，`clamp(logΛ, −7, 7)`；**末层 weight 与 bias 零初始化**
  （初始 logΛ≡0 ⇒ Λ≡1，铁律 6）。隐宽 64 为默认（config 可调）。
  - norm 选 **GN**：BN 在"逐 micro-batch 整批单模式 + batch=2"下统计量随模式震荡（高危，否决）；
    无 norm 为次选开关。
  - **输入默认投影前原始分支特征**（F_C 80ch / F_L 256ch）：输入是冻结特征、分布稳定，
    Λ 评估"原始模态证据质量"；若用投影后特征，proj 缩放与 Λ 存在互相补偿的退化解。
    预埋 `lambda_input='raw'|'projected'` 开关（开放问题 I-6）。
- **PoE 闭式融合**：`μ_F = (Λ_C·P_C(F_C) + Λ_L·P_L(F_L)) / (Λ_C + Λ_L + eps)`，eps=1e-6（铁律 6）；
  Λ_m 形状 [B,1,H,W] 广播到 D 通道；输出 [B,256,180,180]（✅ P3 实测 fusion 出 256ch/180²）直接喂 pts_backbone。
- forward 返回 dict：`{mu_F, log_lambda_C, log_lambda_L, pc, pl}`（pc/pl=投影后分支特征，供 L_teach）。
- **PoE 与 L_teach 一律 fp32 计算**（局部 `autocast(enabled=False)` + 显式 .float()，见 I-10）。

### B.2 EPFusion（ep_fusion.py）挂载与调用关系

- `__init__`：`super().__init__(...)` 照常构建 `self.fusion_layer = ConvFuser`
  ——**属性名保留给冻结教师**，`load_from` 的 ckpt key `fusion_layer.*` 免费命中；
  新建 `self.poe_fuser = PoEFuser(...)`（R0 模式则另建可训练 `self.student_fuser = ConvFuser` 副本，
  权重拷贝时机见 F 章）；随后对冻结集合统一 `requires_grad_(False)`。
  ckpt 加载：mmengine `load_from` 非严格加载，`poe_fuser.*` 仅 missing 警告（恰用零/随机初始化）
  ——加载行为由探针 P2/P6 佐证。备选"init_cfg/load hook 重映射 key"复杂无收益，否决。
- **override `extract_feat`**（结构参照父类 bevfusion.py:240-284 在新文件重写，铁律 11 不破）：
  1. 冻结分支前向整体包 `torch.no_grad()`（免建图，显存红利已由 P4 证实——冻结主干后
     可训练仅 5.61M/冻结 35.19M，bs2+AMP 峰值仅 4.70GB，显存极宽裕，详见 F 章）；
  2. 父类 extract_img_feat 拆成两个子方法：`_img_feats_2d(imgs)`（backbone+neck+reshape，
     对应父类 :141-151）与 `_img_bev(feats_2d, points, ...)`（fp32 autocast 包 view_transform，
     对应 :153-163）——**corrupt_lidar 模式复用 2D 特征、只重跑 view_transform 的前提**（C 章）；
  3. 学生融合调 `self.poe_fuser`（R0 调 `self.student_fuser`）；教师 `self.fusion_layer` 仅 loss 路径调用；
  4. pts_backbone → pts_neck 照旧；`predict()` 继承父类自动走新 extract_feat。
- **override `train(self, mode=True)`**（铁律 10；先例 mmdet3d/models/detectors/imvotenet.py:185-198）：
  `super().train(mode)` 后强制 `img_backbone / img_neck / view_transform / pts_voxel_encoder /
  pts_middle_encoder / fusion_layer` 逐一 `.eval()`（ConvFuser 含 BN，铁律 1 核心对象）。
- override `loss()`：见 E 章。`parse_losses` 直接继承父类（bevfusion.py:75-113）。

## C. 教师特征获取

**机制：子类内直接调用；否决 forward hook。** 理由：(a) 教师需按模式选择性重算半条分支并复用
中间特征，hook 只能旁路截获、表达不了"换输入重跑"；(b) 直接调用数据流显式、可断言；
(c) hook 在 DDP/AMP 下有触发顺序与 autocast 上下文的已知坑，且无收益。

三模式前向流（学生路径中冻结部分与教师路径全部在 no_grad 下；mode 来自
`batch_inputs_dict['corrupt_mode']`）：

```
[clean]  （零额外前向）
  img ── backbone+neck ──> feats_2d ── VT(clean pts) ──> F_C ─┐
  pts ── voxelize+middle_enc ─────────────────────────> F_L ─┤
   学生: poe_fuser(F_C, F_L) → μ_F → pts_backbone → pts_neck → bbox_head
   教师: F_T = fusion_layer([F_C, F_L]).detach()      ← 复用学生自身干净特征；
         分支冻结 ⇒ 与原 BEVFusion 干净前向数值一致（G-1 受控 allclose 验证）

[corrupt_cam]（额外前向 = 干净相机 2D + 干净 VT）
  img_corrupt ── backbone+neck ──> feats_2d' ── VT(clean pts) ──> F_C'  ┐ 学生
  pts(clean)  ── 点云分支 ────────────────────────────────────> F_L    ┤ 学生/教师共享
  img_clean   ── backbone+neck ──> feats_2d  ── VT(clean pts) ──> F_C   ┘ 教师
   教师: F_T = fusion_layer([F_C, F_L]).detach()
   注：本模式点云未损，学生/教师 VT 的深度输入同为 clean points，仅 2D 图像特征不同。

[corrupt_lidar]（额外前向 = 干净 VT + 干净点云分支；2D 特征只算一次复用）
  img(clean) ── backbone+neck ──> feats_2d        （仅一次）
   学生: feats_2d ── VT(pts_corrupt) ──> F_C'；pts_corrupt ── 点云分支 ──> F_L'
   教师: feats_2d ── VT(pts_clean) ───> F_C ；pts_clean  ── 点云分支 ──> F_L
   教师: F_T = fusion_layer([F_C, F_L]).detach()
   ★ 静态勘察修正落点：bevfusion.py:266 将 deepcopy(points) 传入 DepthLSSTransform 造稀疏
     深度图（相机分支并非纯相机）→ 损坏点云会污染学生相机 BEV；教师必须用 clean points
     重跑 VT；2D 图像特征不变可复用（B.2 拆 _img_feats_2d/_img_bev 的原因）。
```

- **detach 位置**：F_T 在 fusion_layer 输出处**立即显式 .detach()**（虽整段在 no_grad 内，
  仍显式 detach 防未来重构破坏；教师侧零可学习变换，投影只在学生侧——铁律 2）。
- **L_teach 精确定义（逐分支高斯 NLL，丢常数项）**：

  `L_teach = Σ_{m∈{C,L}} mean_{b,d,h,w} [ 0.5·Λ_m(b,h,w)·(P_m(F_m) − F_T)²(b,d,h,w) − 0.5·log Λ_m(b,h,w) ]`

  聚合：对通道 d 与格点 h,w 与 batch 全 mean（−0.5·logΛ 随通道广播，与残差项权重一致）。
  **干净样本同样计算**（铁律 7："小误差→高 Λ"是校准的另一半）。
  - 选逐分支而非融合式（以 Λ_C+Λ_L 为精度对 μ_F 算 NLL）的理由：逐分支把 Λ_m 直接锚定到
    "本分支距教师融合特征的距离"，与 PoE 权重语义一一对应、梯度无歧义；融合式两 Λ 可互相
    补偿，存在单边坍缩退化解，且不给 proj 提供逐分支对齐信号。已知副作用：相机分支重建
    F_T 的残差地板更高 ⇒ Λ_C 稳态系统性低于 Λ_L——与 BEVFusion LiDAR 主导的事实相符，可接受。
  - 可选锚定旁路（默认关）：`w_fused_mse · mean‖μ_F − F_T‖²`（config 可调，默认 0），
    若 R1 早期 L_det 发散再启用（开放问题 I-7）。

## D. 退化管线

### D.1 corruptions.py：注册表与严重度

`TRAIN_CORRUPTIONS = {'img': {...}, 'pts': {...}}`，条目为 `fn(tensor, s, gen) -> tensor`，
纯 torch/numpy 手写（铁律 15）。连续严重度 s∈[0,1]；下表区间均为 TUNABLE 默认值，
全部经 config/`--cfg-options` 可覆盖（铁律 17）：

| 条目 | s → 参数映射（默认） | 备注 |
| --- | --- | --- |
| img.gaussian_noise | σ = 50·s（0-255 刻度） | 归一化前 float 原图（铁律 3） |
| img.brightness_contrast | 亮度偏移 ±64·s；对比度因子∈[1−0.6s, 1+0.6s] | 复合后 clamp [0,255] |
| img.downsample_blur | 因子 f = 1+3s（双线性下采再上采回原尺寸） | sanity_p1 主用 |
| img.occlusion_patches | 块数 round(8s)，边长 ≤0.3s·min(H,W)，填均值 | 逐相机独立 |
| img.zero_image | **整路置零特殊条目；severity 无关** | 见 D.2 |
| pts.random_drop | 丢弃率 0.8s（至少保留 20% 点） | |
| pts.xyz_jitter | σ_xyz = 0.2s 米（仅 dim0-2） | |
| pts.beam_drop | 丢 round(24s)/32 线 | ✅ P5 实测第 5 维全 0 → **ring 不可用，正式采用俯仰角 arctan(z/√(x²+y²)) 32 分箱整 bin 丢弃**（删 ring 分支或仅保留回退；多 sweep 聚合云上为近似，注释说明） |
| pts.intensity_noise | σ = 0.2s·强度 robust scale | 强度=dim3；本轮 P5 仅确认第 5 维(ring)，未单独回填 dim3 刻度 → 实现时按样本内 intensity 的鲁棒尺度（分位距/std）自适应，免硬编码 |
| pts.zero_points | **整路置零特殊条目；severity 无关；保留 0.5% 随机点** | 防 voxelize/稀疏卷积空输入崩溃 |

**整路置零 = 注册表特殊条目**（'zero_image'/'zero_points'），隶属对应 corrupt_* 模式
→ 与普通损坏走同一路径，**自动携带干净副本、教师在置零模式同样吃干净输入**。

**TEST_CORRUPTIONS_DOC**：仅模块级注释字符串，列雾/雨/雪/运动模糊/眩光的定义出处与实现要点；
**无任何实现代码，训练代码物理上无法 import 测试族**（铁律 5）。

**损坏 rng**：由 (seed, rank, iter) 确定性派生 `torch.Generator`：
`gen.manual_seed(seed*1000003 + rank*9973 + iter)` —— 跨 rank 去相关、整 run 可复现（铁律 9）。

**M1 前置条款（训练/测试族查重审计）**：进入 M1 的 nuScenes-C 评测前，必须对照
Dong et al.（Benchmarking Robustness of 3D Object Detection, nuScenes-C）完整损坏清单
逐项查重本训练族。已知嫌疑：img.gaussian_noise、brightness_contrast vs nuScenes-C 的
Gaussian/Brightness/Contrast；pts.random_drop/beam_drop vs Incomplete Echo/Beam Missing。
重叠项处理二选一：(a) 替换训练实现（如高斯噪声→均匀/斑点噪声）；(b) 保留但把对应
nuScenes-C 项移入 seen-family、评测时单独报告。M0 不跑 nuScenes-C 故不阻塞，审计结论
须记入 EXPERIMENTS.md 后方可启动 M1 评测。

### D.2 模式采样与干净副本机制（选定候选 2：preprocessor 子类）

**选型**：`EPFusionDataPreprocessor(Det3DDataPreprocessor)` 在 GPU、归一化前注入。理由：
(a) 候选 1（pipeline transform）需绕 Pack3DDetInputs 的 INPUTS_KEYS=['points','img']
（formating.py:50），要三处新类，且 CPU 损坏拖慢 dataloader、干净副本走 collate 翻倍传输；
(b) 候选 2 天然满足铁律 3 两个插入点：图像损坏在归一化（data_preprocessor.py:224/280）之前，
点云损坏必然在体素化（模型内 bevfusion.py:170）之前；几何增广在 pipeline 早已完成，顺序正确；
(c) 干净副本 = GPU 上 clone，零搬运。

`simple_process(data, training)` 流程（仅 training=True 启用；val/test 永不损坏，除非 force_*）：
1. cast 上 GPU；
2. 逐 micro-batch（每卡 batch=2）**整批同一模式**采样：clean 50% / corrupt_cam 25% /
   corrupt_lidar 25%——教师额外前向可整批张量化；每优化步 4 卡×累积 4 = 16 个 micro-batch，
   模式在有效 batch 内充分混合；
3. 损坏模式内以 **p_zero=0.05** 概率抽中置零条目（'zero_image'/'zero_points'；教 Λ 走向
   clamp 下界的极端校准；更高则学生 BEV 编码器频繁吃退化融合特征、L_det 不稳），
   其余概率均匀抽**一种**普通损坏（单一损坏便于 Λ-严重度归因），s ~ U(0.1, 1.0)（可调）；
4. clone 干净副本 → 施损 → 走父类 collate/归一化路径；干净副本同一 preprocess 归一化后按需挂载：
   `inputs['imgs_clean']`（仅 corrupt_cam）/ `inputs['points_clean']`（仅 corrupt_lidar）；
5. 模式经 `inputs['corrupt_mode']`（字符串）传入模型，EPFusion.loss() 读取
   （batch 级信息不放 sample 级 metainfo，语义错位）；
6. 构造参数 `force_mode / force_corruption / force_severity`（默认 None），供诊断脚本经
   `--cfg-options model.data_preprocessor.force_mode=...` 在推理期强制损坏。

## E. 损失与训练

**EPFusion.loss()**：
```
losses = self.bbox_head.loss(feats, batch_data_samples)        # L_det（key 含 'loss' → 反传）
losses['loss_teach']    = w_teach * L_teach                     # 含 'loss' → 反传（parse_losses:102）
losses['teach_nll_raw'] = L_teach.detach()                      # 仅日志，跨 w_teach run 可比
losses[<Λ 与坍缩预警键>]                                        # 仅日志，见下
```
- 干净样本同样计入 L_teach（铁律 7）；w_teach 为 model config 字段（R1-R3 经 --cfg-options 切换）。
- **Λ 日志（铁律 8 + all_reduce 死锁规避）**：父类 parse_losses（bevfusion.py:106-111）对每个
  key 做 dist.all_reduce → 各 rank key 集合必须恒定。方案：每步恒定输出 12 个 key
  `lambda_{C,L}_{clean,corrupt_cam,corrupt_lidar}_{sum,cnt}`，缺席模式填 0；
  all_reduce 取均值对 sum 与 cnt 同除 world_size、比值不变，真实逐模式均值 = sum/cnt
  由日志后处理脚本计算（NaN-safe、无模式饥饿偏差）。key 用 '_' 不用 '/'。
- **坍缩预警日志**（w_teach=10 哨兵；均仅日志键）：`proj_var_C / proj_var_L`
  （投影特征逐批方差）与 `cos_pc_ft / cos_pl_ft`（投影特征对 F_T 的逐格余弦均值）——
  方差→0 或双余弦同时→1 提示 Λ/投影坍缩。
- **可训练集合** = {poe_fuser（含 proj 与 Λ 头）, pts_backbone, pts_neck, bbox_head}
  （后三者由 load_from 自复现 ckpt 初始化）；**冻结集合** = 其余全部（相机分支、
  pts_voxel_encoder、pts_middle_encoder、教师 fusion_layer）：requires_grad_(False) +
  train() 强制 .eval() 双保险（铁律 1/10）。
- **优化器分组**（paramwise_cfg 先例 configs/groupfree3d/...py:199-208）：
  ```python
  optim_wrapper = dict(paramwise_cfg=dict(custom_keys={
      'poe_fuser':    dict(lr_mult=1.0),   # 新模块 lr = 2e-4
      'pts_backbone': dict(lr_mult=0.1),   # 解冻预训练部分 lr = 2e-5
      'pts_neck':     dict(lr_mult=0.1),
      'bbox_head':    dict(lr_mult=0.1)}))
  ```
  ✅ 探针 P6 实测：requires_grad=False 参数被优化器跳过（weight_decay=0.5 放大下 3 步更新
  仍位级不变 = PASS）→ 双保险充分，无需为冻结模块额外补 lr_mult=0；paramwise_cfg 的
  custom_keys lr_mult 也经 P6.6 核验生效（lr=2e-5 与 2e-4 两档如期出现）。
- **轮数：建议 3 epochs**（≤6）：主干冻结，从零学的只有小容量 Λ 头与 1×1 投影；
  pts_backbone/neck/head 自已收敛 ckpt 以 0.1×LR 微调。`--cfg-options train_cfg.max_epochs=`
  可调（一次 zip 往返内可加训）。**param_scheduler 端点须同步覆盖**：LinearLR warmup 500 iter
  保留；CosineAnnealingLR T_max=3/end=3；两段 CosineAnnealingMomentum 端点按比例缩放为
  0–1.2 / 1.2–3（漏改会越界——sanity_lambda_logging 打印首步 LR/momentum 核对）。
- AMP：照旧 `--amp`（tools/train.py:93-105 → AmpOptimWrapper, loss_scale='dynamic'）；
  accumulative_counts=4 与 `--sync_bn torch` 保留（SyncBN 作用于可训练的 pts_backbone/neck BN；
  冻结 BN 因强制 eval 不受影响）。L_teach 与 PoE 强制 fp32（I-10）。

## F. 运行矩阵与估时

| Run | 配置 | 目的 |
| --- | --- | --- |
| R0 | 冻结主干 + **可训练 ConvFuser 副本**（student_fuser，自教师权重拷贝初始化，lr_mult=0.1）+ pts_backbone/neck/head 可训练 + **同样的退化演练数据流** + 同 schedule；无 Λ/无 L_teach | 公平锚点：与 R1-R3 唯一差别 = 融合机制 + L_teach，把"退化演练增强本身的鲁棒性收益"从机制收益中剥离（可选 R0'：无退化纯净版，默认不跑） |
| R1 | EPFusion, w_teach=0.1 | 弱教师 |
| R2 | EPFusion, w_teach=1 | 中点（先跑） |
| R3 | EPFusion, w_teach=10 | 强教师（坍缩预警重点盯） |

- **R0 学生 fuser 权重拷贝时机（关键正确性点）**：拷贝必须发生在 checkpoint 加载之后。
  选型 = 自定义 **`R0WeightCopyHook`**（epfusion/hooks.py，注册进 R0 config custom_hooks，
  `before_train` 时机）。时机论证：mmengine `Runner.train()` 中 `load_or_resume()` 先于
  `train_loop.run()`，而 `before_train` 是 loop.run() 内首个 hook 点 ⇒ before_train 必然晚于
  load_from 权重加载。✅ **探针 P6 实证：Runner.train 中 load_or_resume(L63) 早于
  train_loop.run(L75) → before_train 时机方案成立**（机器判定 PASS）。
  **兜底**：万一未来 mmengine 改顺序，改为 EPFusion 内首步惰性拷贝（首次 loss() 调用时检查标志位执行）。
  **resume 守卫**：Hook 仅在 `runner.iter == 0` 时执行拷贝（resume 续训时学生权重已训练，
  严禁覆写）；拷贝与跳过两条路径**各打印一行**带 run 信息（iter/epoch/VERSION）的日志。
- R0 单独 config 文件；R1-R3 共用主 config 仅改 `--cfg-options model.w_teach=`。
- **估时（✅ 探针 P4 实测回填）**：
  - 显存（冻结主干，可训练 5.61M / 冻结 35.19M，峰值 reserved）：bs1+AMP **2.52GB**、
    bs2+AMP **4.70GB**、bs2 非AMP **4.95GB** → 24GB A30 **极宽裕**；线性外推 **batch 可设 4**
    （≈9GB），仍远低于卡容量。
  - 走时（稳态/iter）：bs2+AMP **0.50s**、bs2 非AMP **0.64s** → **AMP 比非AMP 快 ~20%，
    沿用 `--amp`**。
  - 单 epoch（28130 样本，CBGS 后）：bs2 单卡 ≈ 14065 iter × 0.5s ≈ **1.95h/单卡**；
    **3 卡 DDP ≈ 40–45min/epoch**。3 epochs ≈ **2–2.5h/run**（R0/baseline 形状）。
  - ⚠️ 注意：P4 测的是"冻结主干 + 训 BEV encoder/头"的 R0 形状（stock BEVFusion，无教师
    额外前向）。R1-R3 的 EPFusion 每 iter 还要加 corrupt_cam/corrupt_lidar 模式下的教师
    干净前向（各 ~25% 批次），实际 iter 走时会高于 0.50s，**单 run 估时上浮**——以首个 EP run
    实测为准，回填 EXPERIMENTS.md。batch 设 4 可部分抵消（吞吐↑）。
- 每 run：`randomness=dict(seed=2026)`（R0-R3 同 seed 保证数据序可比）；启动前更新 VERSION
  （日期 + run 名 + 部署 SHA）；训练命令与 work_dir 名记入 EXPERIMENTS.md（铁律 9/16）。
- 全部训练命令从全新解压目录出发：
  ```bash
  bash scripts/deploy.sh --data <nuscenes> --work-dirs <work_dirs> --ops-from <旧部署目录>
  bash tools/dist_train.sh projects/EPFusion/configs/epfusion_m0_poe_4xa30-amp-accum_nus-3d.py 4 \
      --amp --sync_bn torch \
      --cfg-options load_from=work_dirs/baseline1/epoch_19.pth model.w_teach=1.0 randomness.seed=2026
  ```

## G. 诊断与验收脚本

通用约定（铁律 13/14/15/17/18）：每脚本一个 CLI 入口、零新依赖、跑前自检无训练进程、
全部结论汇入单一报告 `<脚本名>_<YYYYMMDD>.txt|.json`（可附 PNG）、报告头记 VERSION 与
ckpt 路径；统一参数 `--config --checkpoint --data-root --out-dir` + `--cfg-options` 透传。

1. **sanity_lambda_logging.py**（块 1 验收闸门）：构建模型 + 1 个真实 train batch，
   分别强制三模式各跑一次 loss()+parse_losses，断言：
   (a) 12 个 Λ key + loss_teach/teach_nll_raw + 坍缩预警键全部在场；
   (b) 初始化（仅 load_from + poe_fuser 零初始化）下 mean(Λ)≡1（零初始化是精确的，容差 1e-6）；
   (c) train pipeline cfg 无 'ObjectSample'（铁律 4）;
   (d) **教师路径受控 allclose 一致**：EPFusion 与原 BEVFusion 加载同一 ckpt、双 eval、
       强制 fp32、同一固定 batch 下，EPFusion 内部 F_T 与原模型 fusion_layer 输出
       `torch.allclose(rtol=1e-4, atol=1e-5)`；
   (e) **R0 权重拷贝断言**：学生 student_fuser 与教师 fusion_layer 权重逐张量相等——
       断言前须在 load_checkpoint 之后**显式调用 R0WeightCopyHook 的拷贝函数**
       （本脚本不经 Runner.train()，before_train 不会自动触发）；
   (f) **单卡 backward 冒烟**：一次 loss 反传后，全部可训练参数 grad 非 None
       （且抽查冻结参数 grad 为 None）；
   (g) 'zero_image'/'zero_points' 强制路径冒烟（不崩溃、Λ 日志在场）；
   (h) 打印首步 LR/momentum（核对 param_scheduler 端点与 max_epochs 同步）。
   报告 `sanity_lambda_logging_<日期>.txt`。
2. **sanity_p1.py**：`--num-frames 50 --corruption downsample_blur --severities 0,0.2,0.4,0.6,0.8,1.0`。
   仅用 TRAIN 族损坏（铁律 5，**禁用 nuScenes-C 损坏**）；50 个 val 帧逐严重度强制 corrupt_cam
   （经 preprocessor force_*），抽 Λ_C/Λ_L 全图均值，输出均值-严重度曲线 PNG
   （matplotlib ✅ P1 白名单确认在列，版本 3.5.3）。**量化判据**：
   Λ_C：Spearman ρ ≤ −0.9 且置换检验 p < 0.05（numpy 手算，1000 次置换）；
   Λ_L：|ρ| < 0.5 **或** 其相对变化 < Λ_C 相对降幅的 20%。
   报告 `sanity_p1_<日期>.json` + PNG。
3. **sanity_p4.py**：`--checkpoint --checkpoint-r0 --modality cam|lidar --num-frames N`。
   固定 val 子集上经 force_mode 强制单模态置零（复用 'zero_*' 条目），分别对 EP 与 R0
   跑 predict + NuScenesMetric 子集评测，报告两者 clean→zeroed 退化幅度与差值。
   报告 `sanity_p4_<日期>.json`。
4. **eval_quarter_val.py**：`--fraction 0.25`，用 mmengine BaseDataset `indices`（每 4 帧取 1）
   做 1/4 val 快评（Runner.test()），输出 NDS/mAP。R0-R3 中期对比统一用此；终选 run 跑全量 val。
   报告 `eval_quarter_val_<日期>.json`（含 ckpt 路径、VERSION、seed）。

## H. M0 验收判据

1. **clean 精度（相对差判据 + 绝对值锚）**：
   - 相对差：EP 最优 run 的 clean NDS ≥ **R0 − 1.0**（注意：R0 = 冻结对照组，**不是**全量复现 run；
     EP 与 R0 同口径评测——同 eval_quarter_val 或同全量 val）。
   - **绝对值锚**：R0 与 EP 的 clean NDS 必须同时报告**相对探针 P2 复现参照值的差**；
     若两者均低于参照 **2.0 NDS 以上** → 触发损坏配比/训练长度排查（怀疑退化演练配比过强或
     轮数不足），**相对差判据暂停适用**，排查收敛后重新评估。
2. **P1 校准曲线（量化）**：Λ_C 满足 Spearman ρ ≤ −0.9 且置换检验 p < 0.05（单调下降）；
   Λ_L 满足 |ρ| < 0.5 或相对变化 < Λ_C 相对降幅的 20%（平坦）。
3. **P4 鲁棒性**：单模态置零下 EP 的 NDS 退化幅度显著小于 R0（同子集同口径）。
   **注记：P4 验证的是机制有效性（PoE 按 Λ 正确降权失效模态），不是零样本泛化；
   零样本泛化（未见损坏/nuScenes-C）的主张属 M1 范围。**
4. **机制 sanity**：初始 Λ≡1（精确）；教师路径受控 allclose 一致（G-1(d)）；
   R0 权重拷贝断言通过（G-1(e)）；三模式 Λ 日志全程在场且无 NaN。

任一失败 → 优先用 I 章预埋的 config 开关零代码迭代（一次 zip 往返为代价单位）。

## I. 风险与开放问题

1. **教师/冻结主干 checkpoint【已决，开放问题闭合】**：锁定
   **`work_dirs/baseline1/epoch_19.pth`**（20 轮 run 暴跌前最佳点，NDS 0.696 / mAP 0.665）。
   理由：(a) 与 R0/R1-R3 同源初始化，锚点公平性最大化；(b) BN running stats 与本机
   SyncBN/有效 batch（4 卡×2）设置同源，教师 eval 输出分布与学生训练分布一致；
   (c) 数据预处理完全同构。**epoch_6（失败的首次 6 轮 run）与 epoch_20（20 轮 run 暴跌点）
   永久弃用**（暴跌为 CBGS 调度端点固有行为，见 RECON §4）。复现 Δ≈−1.8 NDS（< 官方）经
   双 run 自洽 + SyncBN 归因判定为已知系统性差异、非 bug；因 M0 锚点是本机 R0 冻结对照
   （H 章）而非官方分，该差异不影响相对比较，故不阻塞。
2. **共享投影维 D**：**256（= pts_backbone 入口，✅ P3 实测 fusion 入/出 256ch 坐实）**；
   压小 D 需 out-proj，M1 再议。
3. **投影后是否加 norm 及位置**：默认无 norm（线性投影 + 对齐由 L_teach 驱动）；
   **BN 高危否决**（整批单模式 micro-batch 使 BN 统计随模式震荡）；GN 为备选 config 开关。
4. **BEV encoder 全量 vs 部分解冻**：默认 pts_backbone+pts_neck+bbox_head 全解冻（0.1×LR）；
   若 R0 显示 clean 跌幅过大，备选只解冻 pts_neck+bbox_head——预埋 config 开关，零代码迭代。
5. **干净副本携带机制**：候选 2（preprocessor 子类）已选（D.2 给理由）；候选 1 留档不实现。
6. **Λ 头输入投影前 vs 投影后**：默认投影前 raw（B.1 理由）；预埋 `lambda_input` 开关，
   P1 曲线不单调时一次 CLI 切换对照。
7. **L_teach 形式**：逐分支已选（C 章理由）；w_fused_mse 锚定旁路可调（默认 0）；
   融合式 NLL 仅当 R1-R3 出现 Λ 全面坍缩才回头评估。
8. **代码同步与交接**：已闭环（用户拍板）——GitHub 按 commit SHA 下载 zip → 上传离线服务器
   → 解压为全新 SHA 命名目录 → `scripts/deploy.sh`（拷贝 ops *.so / 软链 data+work_dirs /
   打印 VERSION / 环境自检）。服务器无 git/无网/不能 pip install（铁律 15/16）。
   注意：**不要解压覆盖正在使用的旧目录**；迭代代价 = 一次人工 zip 往返，故一切可调项
   走 --cfg-options（铁律 17），交付按"少块大交付、每块自带 sanity"组织。
9. **损坏输入污染冻结 BN？** 不会：requires_grad_(False) + train() 强制 .eval()（铁律 1/10），
   running stats 不更新；G-1(d) 受控 allclose 断言兜底。
10. **AMP 数值域**：exp(±7) ∈ [9.1e-4, 1096.6]，fp16 上界 65504 本身安全；但 NLL 的 Λ·diff²
    与 PoE 分子 Λ·P(F) 乘积可上探 fp16 上界 → **L_teach 与 PoE 融合强制 fp32**
    （局部 autocast(enabled=False) + 显式 .float()，180×180 下开销可忽略）；
    训练中监控 dynamic loss_scale 异常缩频。
11. **batch=2 模式饥饿**：日志侧 sum/cnt 恒定 key 方案解决（E 章）；梯度侧每优化步
    16 micro-batch，期望 8 clean / 4 corrupt_cam / 4 corrupt_lidar，方差可接受。
12. **DepthLSSTransform 深度输入污染**：已在 C 章 corrupt_lidar 流程针对性处理
    （教师重跑 VT、2D 复用；corrupt_cam 下学生/教师 VT 深度输入同为 clean points）。
13. **zip 工作流版本错乱**：VERSION 纪律（每 commit 更新）+ 部署目录 SHA 命名 +
    所有报告文件头记 VERSION/ckpt 路径 + deploy.sh 自检不匹配即终止。
14. **空/极稀点云崩溃**：'zero_points' 保 0.5% 点、random_drop 保 ≥20% 点（损坏函数内
    强制下限）；sanity_lambda_logging(g) 冒烟覆盖。
15. **调度器端点与 max_epochs 不同步**：E 章已列改法；sanity_lambda_logging(h) 打印
    首步 LR/momentum 核对。

### 实施顺序（少块大交付，每块自带 sanity；探针报告回传且 Δ 门通过后启动）

1. **块 1**：epfusion/ 全包 + 两个 config + sanity_lambda_logging.py
   （一次 zip；服务器跑 G-1 全断言通过 → commit 记录结果）。
2. **块 2**：sanity_p1.py / sanity_p4.py / eval_quarter_val.py
   （可与块 1 合并同一 zip；不依赖训练产物即可先行构建自检）。
3. 运行序：R0 → R2（w_teach=1 居中先行）→ 视 R2 的 P1 曲线决定 R1/R3 顺序；
   每 run 结束跑 eval_quarter_val + sanity_p1，结果记 EXPERIMENTS.md。
