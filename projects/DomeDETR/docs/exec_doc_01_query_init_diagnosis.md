# 执行文档 #1：TransFusion 头 query 初始化瓶颈诊断

- 版本：v1，2026-10-16
- 执行者：Claude Code（离线服务器）
- 结果审阅：Claude（对话侧），按第 9 节「预先登记的判读规则」
- 决策者：Hermanuh

## 1. 文档信息与输入

本文档交给 Claude Code 执行：不训练、约 1–2 天，判断 BEVFusion 的 TransFusion 头在 query 初始化环节是否存在足够大的提升空间，以决定 Dome-DETR 迁移方向是否立项。

| 项 | 内容 |
| --- | --- |
| 输入物 | `query_diag.tar.gz`（参考实现，只在 mock 上测过）；BEVFusion 已训练权重；nuScenes val（6019 帧） |
| 开跑前由用户补充 | 配置文件路径：`<CONFIG>`；权重路径：`<CKPT>`；该权重已知的 val mAP / NDS：`<mAP>` / `<NDS>`（第 3 步自检要用） |

## 2. 目标与待验证假设

待验证假设 H0：TransFusion 头在初始化阶段漏掉了相当比例的远距离目标（heatmap top-K 偏向近处强响应），并且这部分漏检对 mAP 的影响足够大。H0 成立是 Dome-DETR 式密度/距离自适应 query 分配有意义的前提。

| 编号 | 回答的问题 | 方法 | 对应决策 |
| --- | --- | --- | --- |
| D1 | 每个 GT 附近有没有初始 query？按距离、类别、query 预算 K 分别统计 | 复现 head 的初始化规则，统计初始 query 召回率；同时统计解码后中心的召回率 | 漏检是否集中在远处；加大或重新分配预算能否找回 |
| D2 | 如果初始化位置全对，mAP/NDS 最多能涨多少 | 测试时把 GT 位置注入 heatmap（oracle），query 预算不变，GT 置信度不泄漏 | 任何只改 query 分配/初始化的方法的性能上限 |
| D3 | 每帧 GT 数量是否会超过 num_proposals（200） | 统计 val 集每帧 GT 数的分布 | "可变 query 数量"能否带来精度收益，还是只能换效率 |

D1 和 D3 在同一次推理里完成；D2 需要 5 次完整的 val 测试。

## 3. 环境与硬约束

以下约束全部是硬性要求，与本文档其他部分冲突时以本节为准。

- **代码基**：mmdet3d 1.4.0 + `projects/BEVFusion`，源码模式运行，所有命令在 mmdetection3d 根目录执行，并设 `PYTHONPATH=$PWD`。
- **算力**：离线、与其他课题组共享的 4×A30。D1 只用 1 卡；D2 默认 1 卡，最多 2 卡。每次开跑前用 `nvidia-smi` 确认目标卡空闲，不得抢占他人进程。
- **不改 baseline**：不修改 `projects/BEVFusion` 下任何源码、配置和权重。诊断逻辑只能放在独立目录（建议 `tools/query_diag/`），通过 forward hook 或实例方法替换实现。
- **同步门禁**：所有 .py 文件同步到服务器前必须通过两项检查：无非 ASCII 字符；`ast.parse` 通过。注释和打印信息一律用英文。
- **先定位再报告**：在部署或运行任何东西之前，先完成第 5 节（第 0 步）源码核对并回传报告；核对通过后再继续。
- **不训练**：本文档内所有步骤只做推理和评测。

## 4. 参考实现说明

`query_diag.tar.gz` 是参考实现，不是已验证的成品。Claude Code 可以直接使用，也可以按第 0 步核对结果修改；任何改动都要在回传材料里说明改了什么、为什么改。

| 文件 | 职责 |
| --- | --- |
| `diag_common.py` | 共享工具：按 head 规则复现初始 query（3×3 局部极大 NMS、nuScenes 第 8/9 类跳过 NMS、跨类全局 top-K）；BEV 格子与米制坐标互转；从 `eval_ann_info` 读 GT；hook 安装；oracle 注入与分数还原 |
| `diag_query_recall.py` | D1 + D3：单卡推理，输出表 T1–T5、`recall_report.txt`、`recall_summary.json`、`recall_per_gt.npz` |
| `diag_oracle_query.py` | D2：在 `bbox_head.heatmap_head` 输出上注入 GT 位置，走配置里的官方 nuScenes 评测；支持单卡或 torchrun |
| `diag_summarize.py` | 汇总 5 组 oracle 结果，计算相对 baseline 的 ΔmAP / ΔNDS |
| `run_diag.sh` | 四个入口：`quick` / `recall` / `oracle` / `summary` |

**已验证**（CPU、mock head，mock 按 mmdet3d 1.4 的 `forward_single` 逻辑编写）：初始化复现与 head 输出逐项一致；坐标约定与 `get_targets` 一致（格心误差 ≤0.3 m）；oracle 注入的格子全部进入 top-200，分数还原正确；命中统计正确。

**未验证**：与 mmengine Runner 的集成；用户仓库里真实的 `transfusion_head.py`；测试流水线里是否带 `eval_ann_info`；多卡（DDP）路径。这几项由第 0 步和自检项覆盖。

## 5. 第 0 步：源码核对（只读，先报告后动手）

只读不改。在用户的仓库里逐项找到下表的代码位置，回传一张核对表：编号、文件路径与行号、原文片段（≤10 行）、与参考实现假设是否一致、不一致时的适配方案。**A1、A3、A4、A7 任一项不一致，先停下报告，等确认后再改。**

| 编号 | 核对位置 | 参考实现的假设 |
| --- | --- | --- |
| A1 | `TransFusionHead.forward_single` 的 query 初始化 | heatmap 取 sigmoid；`nms_kernel_size`×`nms_kernel_size` 局部极大，边缘 padding 区为 0；`dataset=='nuScenes'` 时第 8、9 类不做 NMS；跨所有类别 `argsort` 取前 `num_proposals`；类别 = 索引 // (H·W)，格子 = 索引 % (H·W) |
| A2 | 同上 | `self.query_labels` 在 forward 中被赋值为初始化类别 |
| A3 | `create_2D_grid` 与 `self.bev_pos` | meshgrid 为 ij 顺序并加 0.5；展平索引 r·W+c 对应 (r+0.5, c+0.5)，r 对应 x 轴 |
| A4 | `BEVFusion.predict` → `bbox_head.predict` → `predict_by_feat` | head 通过 `self(...)` 或 `self.forward(...)` 调用；最终分数 = 预测 heatmap 的 sigmoid × `query_heatmap_score` × one_hot(`query_labels`)，即 query 类别在初始化时锁定 |
| A5 | `heatmap_head` 的调用 | 每次 forward 只调用一次，以模块形式调用（hook 能触发），输出为 float32 logits，形状 [B, 10, 180, 180] |
| A6 | 配置中的 `test_cfg` 与 `num_proposals` | 含 `dataset`、`out_size_factor`=8、`voxel_size`=[0.075, 0.075]、`pc_range`=[-54, -54]；`num_proposals`=200 |
| A7 | 测试流水线与 `Pack3DDetInputs` | `data_sample.eval_ann_info` 存在，含 LiDAR 坐标系下的 `gt_bboxes_3d` 和 `gt_labels_3d`；测试时没有对点云做旋转、缩放、平移等增强 |
| A8 | `get_targets_single` 画高斯处 | 使用 `center_int[[1, 0]]` 的修正写法，即 heatmap 行号对应 x |
| A9 | 配置中的 `class_names` | 索引 8 = pedestrian，9 = traffic_cone |

## 6. 第 1 步：部署、门禁与 300 帧冒烟

第 0 步通过后，部署脚本、跑门禁，再用 300 帧冒烟。S1–S4 全部通过才能进入第 2 步；任一项失败就停下，回传日志，不要为了通过自检去改统计口径。

1. 把 `query_diag/` 放到 `tools/query_diag/`，对其中所有 .py 文件跑非 ASCII 检查和 `ast.parse`，对 `run_diag.sh` 跑 `bash -n`。
2. 运行冒烟（单卡）：

   ```bash
   CONFIG=<配置路径> CKPT=<权重路径> CUDA_VISIBLE_DEVICES=<空闲卡> \
     bash tools/query_diag/run_diag.sh quick
   ```

3. 记录耗时和单卡峰值显存，用来估算全量运行时间。

| 自检 | 判据 | 失败说明什么 |
| --- | --- | --- |
| S1 | 报告首行 `init vs head.query_labels match rate` ≥ 0.99 | 你的初始化逻辑与复现规则不同（回到 A1） |
| S2 | 日志打印 `bev_pos transposed=False` | 坐标约定与假设不同（回到 A3） |
| S3 | 报告中无 `[WARNING]`；0–20 m 段 class-aware 初始召回率 @2 m 写入回传（预期明显高于 0.5，低于 0.5 即停） | 坐标或 GT 读取有误（回到 A3、A7） |
| S4 | T4 中 0–20 m 段 `final_agn` ≥ `init_agn` − 0.05 | 解码器 `center` 的单位换算有误 |

## 7. 第 2 步：全量 D1 + D3

在完整 val 集（6019 帧）上单卡运行一次，参数固定为下表默认值，不得为了结果调整。

```bash
CONFIG=<配置路径> CKPT=<权重路径> CUDA_VISIBLE_DEVICES=<空闲卡> \
  bash tools/query_diag/run_diag.sh recall
```

| 参数 | 固定值 |
| --- | --- |
| query 预算 K | 200、300、500、1000 |
| 中心距离阈值 | 0.5、1、2、4 m（主阈值 2 m） |
| 自车距离分段 | 0–20、20–30、30–40、40–50 m |
| nuScenes 按类别评测距离截断 | 开启 |

**额外汇总（用 `recall_per_gt.npz` 计算，不改主脚本）**：增加 "<30 m" 和 "≥30 m" 两个合并段，每段给出四个数：

1. class-aware 召回率 @2 m，K=200
2. class-aware 召回率 @2 m，K=1000
3. class-agnostic 召回率 @2 m，K=200
4. 解码后 `final_ca` @2 m

第 9 节的规则直接用这些数。

产出物：`work_dirs/diag/recall/` 下的 `recall_report.txt`、`recall_summary.json`、`recall_per_gt.npz`，以及上述两个合并段的数。

## 8. 第 3 步：D2 oracle 五组

五组按顺序串行运行，每组是一次完整的 val 推理加官方 nuScenes 评测。`base` 组先跑，S5 不通过就停止，后四组不跑。

```bash
CONFIG=<配置路径> CKPT=<权重路径> GPUS=<1 或 2> \
  bash tools/query_diag/run_diag.sh oracle
bash tools/query_diag/run_diag.sh summary
```

| 组 | 参数 | 回答的问题 |
| --- | --- | --- |
| base | `--oracle none` | 挂上 hook 后能否复现原始结果 |
| all | `--oracle loc --range 0 1000` | 所有 GT 位置都初始化正确时的上限（类别仍由网络决定） |
| far | `--oracle loc --range 30 1000` | 只修正 ≥30 m 目标的初始化时的增益 |
| near | `--oracle loc --range 0 30` | 只修正 <30 m 目标的初始化时的增益 |
| all_cls | `--oracle loc_cls --range 0 1000` | 位置和类别都给真值时的上限 |

所有组的分数模式都是 `--score orig`：被注入的 query 使用该格子原始的 heatmap 分数，不泄漏 GT 置信度。

| 自检 | 判据 |
| --- | --- |
| S5 | `base` 组的 mAP、NDS 与该权重已知值相差都不超过 0.1 个百分点 |
| S6 | 四个 oracle 组的 `init_mismatch_rate` ≤ 0.01，`capped_cells` = 0（不为 0 时如实报告数量） |
| S7 | 四个 oracle 组的 `restored_queries` ≥ 0.99 × `injected_cells` |

## 9. 预先登记的判读规则

下列阈值在第 3 步开跑前由 Hermanuh 确认，确认后不得因结果修改；审阅时按下方顺序逐条判断。

**定义**：ΔmAP_X = 第 3 步 X 组的 mAP 减去 base 组的 mAP，单位为百分点（pp）。R_far(K) 和 R_near(K) 分别是第 2 步 "≥30 m" 和 "<30 m" 合并段在预算 K 下的 class-aware 召回率 @2 m；miss = 1 − R(200)。

| 规则 | 判据（建议阈值） | 含义 |
| --- | --- | --- |
| R1 总上限 | ΔmAP_all ≥ 1.0 pp（**待确认：1.0 或放宽到 0.8**） | 不满足：只改 query 初始化最多涨不到门槛，停止 Dome-DETR 主线 |
| R2 远距离特异性 | ΔmAP_far ≥ 0.5 pp，且 miss_far ≥ 2 × miss_near | 不满足：方法可以做，但远距离不能作为主卖点 |
| R3 预算能否找回漏检 | R_far(1000) − R_far(200) ≥ 0.05 | ≤ 0.02：漏检来自 heatmap 本身没有响应，加 query 无用，转向增强特征或 heatmap；0.02–0.05：审阅时讨论 |
| R4 类别锁定 | ΔmAP_all_cls − ΔmAP_all ≥ 1.0 pp，或 ≥30 m 段 class-agnostic 召回率减 class-aware 召回率 ≥ 0.05 | 满足：瓶颈是初始化时类别判错，Dome-DETR 不对症，转向类别解耦的初始化 |
| R5 解码器补救 | ≥30 m 段 final_ca − R_far(200) ≥ 0.05 | 满足：解码器已补回部分漏检，方法设计时要下调对初始化改进的预期 |
| R6 可变数量 | GT 数超过 200 的帧占比 < 0.5% | 满足："可变 query 数量"只能写成效率贡献，不能写成精度贡献 |

1.0 pp 这个门槛的依据：实际方法通常只能拿到 oracle 上限的一部分，ALIGN 在 heatmap 初始化的检测头上只涨了约 0.4 mAP。上限低于 1 pp 时，期刊需要的稳健增益基本拿不到。

**判读顺序**（前一步不通过就停在该步的结论）：

1. 自检 S1–S7 全部通过？否 → **不下结论**：排查失败项后重跑，不看 R1–R6。
2. R1 满足？否 → **停止 Dome-DETR 主线**：query 初始化整体不是瓶颈。
3. R4 触发？是 → **转向：类别解耦的初始化**。
4. R3 预算增益 ≥ 0.05？否 → 增益 ≤ 0.02 时**转向：增强特征 / heatmap**；0.02–0.05 时审阅讨论。
5. R2 满足？
   - 是 → **Go：Dome-DETR 方法设计，以远距离漏检为主线**。
   - 否 → **Go：Dome-DETR 方法设计，但不以远距离为主卖点**，叙事改为召回不足 / 拥挤场景。

R5、R6 不改变是否立项，只影响方法设计。

## 10. 回传材料清单

每一步结束后回传对应材料，不要等全部跑完再一起发。npz 文件体积大，不需要回传，除非审阅时另行要求。

- [ ] 第 0 步：A1–A9 核对表（路径、行号、原文片段、是否一致、适配方案）
- [ ] 第 1 步：门禁输出；冒烟的 `recall_report.txt`；S1–S4 逐项结果；300 帧耗时和峰值显存
- [ ] 第 2 步：全量 `recall_report.txt` 和 `recall_summary.json`；"<30 m" 和 "≥30 m" 两个合并段各自的四个数
- [ ] 第 3 步：5 个 `oracle_*.json`；`summary` 的完整输出；S5–S7 逐项结果
- [ ] 环境信息：torch、mmcv、mmengine、mmdet3d 版本，使用的 GPU 数，各步耗时
- [ ] 相对参考实现的所有改动：改动摘要（diff 摘要即可）和原因

## 11. 禁止事项

- 修改 `projects/BEVFusion` 下的源码、配置或权重。
- 看到结果后调整第 9 节中的阈值或统计口径；也不能为了让自检通过而改口径。自检失败就停下报告。
- 运行任何训练。
- 使用超过 2 张 GPU，或在别人占用的卡上运行。
- 删除或覆盖已有的 `work_dirs` 目录；本次所有输出只写到 `work_dirs/diag/`。
- 第 0 步核对报告未经确认就部署、运行。
- 跳过 `base` 组直接跑 oracle 组。

## 12. 已知局限与风险

- **nuScenes 按类别截断评测距离**：锥桶、路障只评测到 30 m，行人、自行车、摩托车只评测到 40 m。因此 "≥30 m" 段只包含车辆类，以及 30–40 m 的行人和两轮车；远处小锥桶不计入官方 mAP。论文动机的表述要与此一致。
- **自车距离是近似值**：用 LiDAR 坐标系下的 xy 范数代替官方的 ego 距离，偏差约 1 m 量级，对 10 m 宽的分段影响可以忽略。
- **oracle 上限偏保守**：被注入的 query 沿用原始 heatmap 分数，所以测到的是"只修正初始化位置"的上限；同时改善置信度的方法可能超过这个上限。
- **结论的适用范围**：D2 若为负，只能排除"只改 query 分配/初始化"的方法，不能排除增强特征或 heatmap 本身的方法。
- **单一权重**：所有结论只针对本次使用的这一个权重。
