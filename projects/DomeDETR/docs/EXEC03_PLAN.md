# 执行文档 #3 · 实施计划（第 0 步，提请确认）

- 依据：`exec_doc_03_query_localization.md`；源码核对见 `EXEC03_STEP0_SOURCE_AUDIT.md`
- 按 CLAUDE.md"重大改动先出计划，经确认后再写代码"，以及 #3 第 4 节"先报告后动手"：
  - **本次已完成**：T1–T6 核对；第 0a 步的离线分析工具；服务器探针；S0''' 检查；运行入口的 `probe`、`s0ppp`、`0a` 三个步骤；以上各项的 CPU 测试。这些都不涉及模型改动。
  - **等确认后再写**：`projects/QueryLoc` 的模型代码（冻结、D3、M-a、M-b）和第 4.2 / 4.3 节的预演与检查工具。

## 1. 目录与文件

```
projects/QueryLoc/
  queryloc/__init__.py      注册新模块（custom_imports 入口）            [待确认后写]
  queryloc/model.py         QueryLocBEVFusion：冻结与 train() 覆盖       [待确认后写]
  queryloc/head.py          QueryLocTransFusionHead：连续位置、M-a、M-b、
                            L>1 的 get_targets 修复、D3 复制初始化       [待确认后写]
  queryloc/subcell.py       DARK / soft-argmax / 3×3 邻域（纯 torch，
                            0a 与 M-a 共用同一实现）                     [本次已写]
  configs/ql_protocol.py    第 3 节统一协议（继承基础配置）              [待确认后写]
  configs/ql_c0.py ql_d3.py ql_mb.py ql_ma.py                            [待确认后写]
  tools/run_queryloc.sh     入口：probe / s0ppp / 0a（本次）；之后加
                            check-load / rehearse / sm0 / train / test /
                            diag / pair
  tools/queryloc_check.py   S0'''（本次）；之后加 S-T0 / S-T2 / S-T3 / S-T4
  tools/subcell_0a.py       第 0a 步（本次）
  tools/probe_step0.py      服务器探针（本次）
  code_md5.json             新代码的 md5 登记表，S0''' 据此核对
tools/query_diag/           第 7.4 节的"读取实际初始位置"功能（第 1 步前加）
```

## 2. 模型设计（等你确认）

| 编号 | 项 | 方案 | 理由 / 备选 |
| --- | --- | --- | --- |
| P-1 | C0 用哪个 head | **原版 `TransFusionHead`，不改**；只换模型类以实现冻结 | C0 就是"基础配置 + 协议"。D3、M-a、M-b 用 `QueryLocTransFusionHead`；S-T5 的 300 帧对拍检查它在选格和分数上与原版完全一致 |
| P-2 | L > 1 的训练 | 在子类中覆盖 `get_targets`，按样本切 `preds_dict[0]`（审计 T1） | 原代码在 L > 1 时 IndexError；L = 1 时修复版与原版逐元素相同 |
| P-3 | D3 的复制初始化 | 在 head 上注册加载前置 hook：state_dict 缺第 i 层（i ≥ 1）而有第 0 层时用第 0 层补齐，包括 BN 缓冲区 | 对 `load_from`、测试脚本、S-T0 都生效；resume D3 自己的 checkpoint 时不会覆盖 |
| P-4 | 冻结 | `QueryLocBEVFusion(BEVFusion)`：<br>• 构造时对 `img_backbone`、`img_neck`、`view_transform`、`pts_voxel_encoder`、`pts_middle_encoder` 设置 `requires_grad_(False)`；<br>• 覆盖 `train(mode)`，每次调用都把这些子模块设为 `.eval()` | CLAUDE.md 铁律 1 / 10；审计 T6 |
| P-5 | 连续位置的特征采样 | 手写双线性：用 4 次 gather 加权，位置为 (行 = x − 0.5, 列 = y − 0.5)，边界处钳位 | δ = 0 时小数部分恰为 0，结果与原 gather 逐位相同，S-M0 因此可以期望与 E0 一致（只剩运行间非确定性）；用 `grid_sample` 做不到逐位相同 |
| P-6 | M-b 的偏移是否回传检测损失 | **detach**：δ 只用于确定 query_pos 和采样位置，偏移头只由稠密 L1 监督 | 文档写的是"解码器的 center 仍相对 query_pos 回归；其余损失不变"。如果不 detach，center 的 L1 会通过 query_pos 直接流进偏移头，偏移头和解码器的回归就重复了，也偏离了"偏移学习"的设计。原 TransFusion 层与层之间的 query_pos 也是 detach 的 |
| P-7 | M-b 偏移头的数值精度 | 与 heatmap 头一样在 fp32 下计算（autocast 关闭） | 回归目标为 ±3 格，tanh 输出在 fp16 下精度不足 |
| P-8 | M-b 损失的细节 | 预测为 tanh × 3 后的 2 通道（格单位，x、y 顺序同 `bev_pos`）。对每个监督格，L1 = \|Δx − t_x\| + \|Δy − t_y\|，再除以监督格数（即"按监督格数取平均"，两个方向求和）；λ = 0.25。<br>GT 连续格坐标 = (x − x0) / vx / 8，与 `get_targets_single` 相同；目标 = 连续坐标 − (格索引 + 0.5)，截断 ±3。<br>多个 GT 时按欧氏距离归属最近的 GT 中心。使用经过训练流水线后的全部 GT，不做类别评测距离截断 | 文档第 7.1 节没有写死两个方向是求和还是平均，这里取求和；如果你希望取平均，损失等价于减半，请明确 |
| P-9 | M-b 偏移头的学习率 | `paramwise_cfg.custom_keys={'bbox_head.offset_head': dict(lr_mult=10)}` | 文档第 3 节 |
| P-10 | M-a 的训练目标 | GT 所在格（floor）= 1；半径 r 内其他格 = exp(−‖(i + 0.5, j + 0.5) − (x̂, ŷ)‖² / 2σ²)；r、σ = (2r + 1) / 6、`< eps·max` 置零、多 GT 取 max 都与原实现相同 | 文档第 7.2 节 |
| P-11 | M-a 的初始位置 | 训练和推理都用 DARK（3×3、log 域、有限差分、截断 ±0.5、Hessian 非负定或不可逆时退回格子中心、边界格退回），与 0a 用同一函数 | 文档第 7.2 节；0a 与 M-a 共用 `subcell.py` |
| P-12 | 两种方法的不变量 | top-K 选择、`query_labels`、`query_heatmap_score` 只从原逻辑取，不经过新代码 | 由 S-T5 的代码比对和 300 帧对拍核对 |

## 3. 第 3 节协议中需要你确认的细节

第 3 节已经规定的内容：峰值 lr × 0.1（即 2e-5）、cosine 退火到 0、前 500 iter 线性 warmup，优化器、weight decay、裁剪、AMP、累积与原配置相同，数据与增强相同，种子 0 / 1，只评测最后一个 epoch。以下几处原文没有写到：

| 编号 | 问题 | 原训练的做法 | 建议 |
| --- | --- | --- | --- |
| R-1 | momentum 调度 | beta1 从 0.9 降到 0.895，再升到 1.0（`eta_min` 是绝对值，见审计 T5） | **不做 momentum 调度，beta1 固定为 0.9**。复制这个形状会让 beta1 在微调末期趋向 1，一阶矩几乎停止更新，对 1–2 个 epoch 的短微调没有好处 |
| R-2 | warmup 的起始系数 | `LinearLR start_factor = 1/3`，按 iter 计（mmengine 每个 dataloader iter 步进一次，不受累积影响） | 照搬：1/3，500 个 dataloader iter |
| R-3 | cosine 的终点 | `eta_min_ratio = 1e-4` | 按文档退火到 0 |
| R-4 | 卡数与 BN | 4 卡 + SyncBN，每次 BN 统计 8 个样本 | **建议 4 卡 + SyncBN**，与起点权重一致。如果只能用更少的卡，就通过累积把有效 batch 保持在 32（2 卡累积 8，1 卡累积 16），但 BN 统计样本会变成 4 或 2，与原训练不同。可训练部分（fuser、SECOND、FPN、head）都含 BN。由第 4.2 节预演确定，回传后由你拍板 |
| R-5 | 训练中的验证与保存 | 每个 epoch 验证一次、保存一次 | 只在最后一个 epoch 验证和保存（`val_interval = N`，`max_keep_ckpts = 1`），中间不评测，也就无从挑选 epoch |
| R-6 | 确定性 | `randomness = dict(seed=577127641)`，不开 deterministic | `seed = 0 / 1`，`deterministic = False`。推理阶段同权重两次运行就有约 0.07 pp 的差（#2），σ_ft 会把这部分也算进去 |
| R-7 | GridMask | 父配置写了 `max_epoch`，但 `prob = 0.0`，实际不生效 | 不变 |

## 4. 第 0 步要做的检查（第 4.2 / 4.3 节）

| 检查 | 实现 |
| --- | --- |
| S0'''（每次运行前） | `queryloc_check.py s0ppp`（**本次已写**）：<br>• CONFIG、CKPT 的 md5 和 `CFG_OPTS` 与 #2 E0 的 runtime 记录相同；<br>• `projects/BEVFusion` 按 82c2b4e 清单（33 个被跟踪文件）逐个核对，源码有差异或多出 `.py` 判 FAIL，`.md` 有差异判 WARN，口径同 #1 的 S0a；<br>• `tools/query_diag/` 与 #2 第 2 步相同（第 7.4 节的改动加入后改为白名单方式）；<br>• `projects/QueryLoc` 与 `code_md5.json` 一致；<br>• 门禁通过 |
| S-T0 | `check-load`：对每个配置构建模型，把起点权重的键与模型的键直接比对，列出 missing / unexpected / size mismatch。<br>C0 应全部匹配；D3 的 missing 应恰好是 `decoder.{1,2}` 和 `prediction_heads.{1,2}`；M-b 的 missing 应恰好是 `offset_head.*`。<br>加载后再核对：D3 的第 1、2 层与第 0 层逐张量相同；M-b 最后一层的权重和偏置全为 0 |
| S-T2 | `rehearse`：用 hook 在 `before_train`（已加载权重之后）对冻结模块的参数和缓冲区做快照，跑 100 iter（累积 4，即 25 次优化器更新），结束后逐张量比对；同时核对可训练参数量，记录 s/iter（去掉前 20 iter）、峰值显存，并推算单 epoch 的墙钟时间 |
| S-M0 | M-b 零偏移在 val 全量推理（单卡，约 38 分钟），mAP、NDS 与 E0 各相差 ≤ 0.001。另加 50 帧逐位比对：M-b 的 query 特征与原 gather 完全相同 |

## 5. 第 0a 步（本次已写，可以先在服务器上跑）

`run_queryloc.sh 0a`：先做 S0'''，再做 C5 检查，然后单卡跑 val 全量，约 30–40 分钟，输出到 `work_dirs/queryloc/step0a/`。

**计算内容**（按文档第 5.1 节）：

- 对每个初始 query（K = 200），在其类别通道的 sigmoid heatmap 上取 3×3 邻域。
- 计算三种位置：
  - 格子中心；
  - DARK：log 域，梯度 = 中心差分，Hessian 含交叉项，偏移 = −H⁻¹∇，每个方向截断 ±0.5；Hessian 非负定或不可逆时退回格子中心，边界格也退回；
  - 3×3 soft-argmax：以 sigmoid 值为权重。
- 对每个 GT，取按格子中心最近的同类 query（2 m 内），按 d_cell 分为 G0 / G1 / G2，并分 <30 / ≥30 m。

**输出**：

- d_cell、d_dark、d_soft 的均值和中位数；
- 三种位置各自的 class-aware 初始召回 @0.5 m（在全部 200 个 query 上计算）；
- 移近比例（d_dark < d_cell）；
- 退回计数，按原因分（边界、Hessian 非负定），并区分做 NMS 的类与不做 NMS 的两类；
- 第 0b 步是否触发：DARK 使全体召回 @0.5 m 提升 ≥ 0.5 pp。

**自检**：

- S-0a1：格子中心的召回 @0.5 m 应等于 B0 的 0.82792，容差 0.001。两者是同一权重、同一定义，用来核对关联和距离的代码。
- S-0a2：复现的 top-K 与 `head.query_labels` 的一致率 ≥ 0.99。

**第 0b 步**只在触发时实现（测试时把 query_pos 换成 DARK 位置，特征仍取峰值格）。

## 6. 耗时估计（第 4.2 节会实测）

- **全量训练的参照**：4 卡约 1.0 s/iter，每 epoch 15448 iter，约 4.3 小时（训练日志）。
- **冻结后**：前向成本不变，相机分支的反向省掉，粗估 4 卡约 0.6–0.8 s/iter，即约 2.6–3.4 小时/epoch。
  - N = 2 时，5 次微调约需 26–34 小时的 4 卡时间，另加 5 次评测和定位诊断。
  - 如果只用 1 卡，每 epoch 的 iter 数变为 4 倍，大概率超过 10 小时，按规则 N = 1。
- 这些只是推算，以预演实测为准。

## 7. 请确认

1. P-1 到 P-12，重点是 P-6（M-b 的 δ 是否 detach）和 P-8（L1 两个方向求和）。
2. R-1 到 R-7，重点是 R-1（不做 momentum 调度）和 R-4（卡数与 SyncBN）。
3. 第 3 节协议和第 9 节阈值本身（文档写明需要你确认后才能进入第 1 步）。

确认后我写 `projects/QueryLoc` 的模型代码和第 4.2 / 4.3 节的工具，CPU 测试和门禁通过后回传，再在服务器上跑预演、S-T0、S-T2 和 S-M0。

## 8. 现在就可以在服务器上跑的部分（probe、0a）

部署方式同 #2a：新 zip 解压到新建的空目录，**只拷 `projects/QueryLoc/` 和 `VERSION`**。`tools/query_diag/` 不动，S0''' 会核对它与 #2 第 2 步相同。

```bash
SRC=/212022085500129/dome-detr/_src_exec03
mkdir -p $SRC && unzip -q <新 zip 的路径> -d $SRC
NEW=$SRC/mmdetection3d-claude-dome-detr-query-diag
WS=/212022085500129/dome-detr/mmdetection3d-claude-dome-detr-query-diag
cp -r $NEW/projects/QueryLoc $WS/projects/
cp $NEW/VERSION $WS/VERSION
md5sum $WS/projects/QueryLoc/queryloc/subcell.py $WS/projects/QueryLoc/tools/*   # 与 code_md5.json 一致

cd $WS && source tools/query_diag/exec02_env.sh
bash projects/QueryLoc/tools/run_queryloc.sh probe          # 只读，几秒
nvidia-smi
CUDA_VISIBLE_DEVICES=<空闲卡号> bash projects/QueryLoc/tools/run_queryloc.sh 0a   # 约 30-40 分钟
```

回传材料：

- `work_dirs/queryloc/probe_step0_*.txt`
- `work_dirs/queryloc/s0ppp/s0ppp_check.txt`
- `work_dirs/queryloc/step0a/` 下的 `subcell_0a.txt`、`subcell_0a.json`（`subcell_0a_per_gt.npz` 可选）
- `work_dirs/queryloc/logs/` 下的日志

## 9. 本次的开发侧验证

- `tests/test_exec3_cpu.py`：7/7 PASS。
  - DARK 在带交叉项的各向异性 log-二次峰上，恢复格内偏移的误差 < 1e-6 格；真峰超出 ±0.5 时截断并标记；平坦、鞍点、边界三种情况都正确退回；soft-argmax 正确。
  - 0a 在高斯 mock 上端到端运行（mock 新增 `QD_MOCK_GAUSS=1`，在连续 GT 位置画 log-二次峰）：
    - 98% 的 G0 / G1 GT 的 d_dark < 1 mm；
    - 格子中心召回与 D1 在同一批帧上的结果逐位相同（S-0a1，容差 1e-12）。
    - 把行列映射故意改反后，d_dark 的中位数从 0 变为 0.235 m，测试能抓到。
  - S0''' 对本仓库判 PASS：`projects/BEVFusion` 33/33 与清单一致，登记表一致。负例（新代码被改、多出源码文件、`tools/query_diag` 被改）判 FAIL，`.md` 改动判 WARN，`.so` 被忽略。
  - 运行器的步骤顺序、门控、C5 检查和日志都正确；guard 函数与 `run_diag.sh` 逐字相同。
- 回归测试：`test_query_diag_cpu.py` 18/18、`test_exec2_cpu.py` 11/11、`test_exec2a_cpu.py` 6/6，全部 PASS。mock 的改动只在设置了环境变量时生效。
- 门禁：新文件全部通过非 ASCII、Python 3.8 语法和 `bash -n` 检查，flake8 无告警。
