# 执行文档 #2：覆盖收益与定位收益的分离测量

- 版本：v1，2026-10-18
- 执行者：Claude Code（离线服务器）
- 结果审阅：Claude（对话侧），按第 8 节「预先登记的判读规则」
- 决策者：Hermanuh
- 前置：执行文档 #1 v1.1 已完成，判读停在 R3 的"审阅讨论"区间；本文档的修改方案和阈值已于 2026-10-18 由 Hermanuh 确认

## 0. 背景与要做的决定

执行文档 #1 的 oracle 显示，初始 query 位置全部给对时 ΔmAP = +1.875 pp，但这部分增益几乎全部来自 0.5 m 匹配阈值（贡献 +1.881 pp），2 m / 4 m 两项为负。这说明测到的提升空间主要在**初始 query 的定位精度**，而不在**覆盖**（哪些目标有 query）。Dome-DETR 式的 query 分配针对的是覆盖，现有实验没有单独测出覆盖能带来多少 mAP。

本文档不训练，回答三个问题：

| 编号 | 问题 | 方法 | 用途 |
| --- | --- | --- | --- |
| B0 | 解码器能把初始位置的偏差修回多少？ | 用执行文档 #1 第 2 步的 npz 离线计算，不占 GPU | 解释机制，不参与决策 |
| E1 / E2 | 只补上初始化漏掉的目标，mAP 涨多少？ | 只对漏检 GT 注入，已有检出不动；E1 还原原始分数（接近真实），E2 保留高分数（上限） | 覆盖方向是否成立（Dome-DETR 去留） |
| E3 | oracle 增益有多少依赖"放在真值所在的格子"？ | 全部 GT 注入，但位置随机偏移 ±1 格 | C 方向（初始 query 定位精化）的前提是否成立 |

## 1. 输入（与执行文档 #1 第 3 步完全相同）

| 项 | 值 |
| --- | --- |
| 工作区 | `/212022085500129/dome-detr/mmdetection3d-claude-dome-detr-query-diag` |
| CONFIG | `projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_4xa30-amp-accum_nus-3d.py`（md5 须为 `8f49ac79…`） |
| CKPT | 与 #1 相同（md5 须为 `c16536e3…`） |
| CFG_OPTS | 与 #1 相同（Swin 初始化路径） |
| 参考 base | #1 第 3 步：mAP 0.665463 / NDS 0.706154 |
| 参考 ΔmAP_all | #1 第 3 步 all 组：+1.875 pp（E3 的分母） |
| D1 数据 | #1 第 2 步的 `work_dirs/diag/recall/recall_per_gt.npz` |
| 输出目录 | `work_dirs/diag/exec2/` |

## 2. 硬约束

沿用执行文档 #1 v1.1 第 3 节的全部约束（不改 `projects/BEVFusion`、同步门禁、C5 GPU 检查、单卡、不训练、所有终端输出写日志）。另加三条：

- 不得改变"漏检"的定义（第 3 节 M1），也不得改动阈值。
- 除 E0–E3 外，不得增跑其他 oracle 变体；如认为有必要，先提出申请，等确认。
- 所有 ΔmAP / ΔNDS 都相对于本文档的 E0 计算（E3 的分母除外，见第 8 节）。

## 3. 代码改动要求（只在 `tools/query_diag/` 内）

| 编号 | 改动 | 要求 |
| --- | --- | --- |
| M1 | `diag_oracle_query.py` 新增 `--subset {all,missed}`（默认 `all`）和 `--miss-thr`（默认 2.0） | `missed` 的定义：在**未修改**的 heatmap 上按 head 规则取 K = num_proposals 的初始 query，某个 GT（经过按类别评测距离截断和 `--range` 过滤）在 `--miss-thr` 米内没有**同类**初始 query，即为漏检。判定必须复用 D1 的函数（`replicate_init`、`gt_from_sample`、`per_gt_hits` 的 class-aware 分支），在同一次 forward 内逐帧计算。只对漏检 GT 注入，注入方式与 `loc` 模式相同（通道 = 该格原始 logits 的 argmax） |
| M2 | 所有 oracle 模式新增计数 `selected_injected` | 被选中的初始 query 中，(类别, 格子) 落在注入掩码上的个数。orig 和 boost 两种分数模式都要统计；同时记录 `gt_in_range`、`missed_gt` |
| M3 | 新增 `diag_b0_decoder.py` | 读取 #1 的 `recall_per_gt.npz`，输出第 5 节的 B0 表格、S9 和 `N_missed_D1`；写 txt 和 json |
| M4 | `run_diag.sh` 新增入口 `b0`、`exec2` | `exec2` 先跑 E0 并自动判 S5'，不通过即停；通过后依次跑 E1、E2、E3。每次启动前执行 C5 检查。所有输出写到 `work_dirs/diag/exec2/` 和 `work_dirs/diag/logs/` |
| M5 | 汇总 | 输出 E1–E3 相对 E0 的 ΔmAP / ΔNDS、E3 保留率、各 S 项的 PASS/FAIL；用已有的 `oracle_breakdown.py` 给出 E1–E3 各匹配阈值的 AP 贡献和分类别 ΔAP。**不输出 Q 类规则的判定** |

实施后跑同步门禁（非 ASCII、`ast.parse(..., feature_version=(3, 8))`、`bash -n`），回传 diff 摘要。旧模式（`--subset all`）的行为必须与 #1 完全相同。

## 4. 第 1 步：一致性检查与 B0（不占 GPU）

1. **S0'**：CONFIG、CKPT 的 md5 与 #1 相同；`CFG_OPTS` 相同；`tools/query_diag/` 中只有 M1–M5 涉及的文件有改动（列出 md5 变化）；门禁通过。
2. 运行 B0：

   ```bash
   bash tools/query_diag/run_diag.sh b0
   ```

3. 检查 S9（第 7 节）。不通过就停。

## 5. B0 的输出内容

全部基于 K = 200、class-aware，分段为 0–20、20–30、30–40、40–50、<30、≥30、全部。

**表 B0-1：初始召回与解码后召回**

每个分段给出 0.5 / 1 / 2 / 4 m 四个阈值下的 init 召回率和 final 召回率，以及两者之差。

**表 B0-2：解码器修正率**

用"最近的同类初始 query 距离"把 GT 分成三组，统计解码后中心进入更严阈值的比例：

| 组 | 定义（用 `rank_ca`） | 统计量 |
| --- | --- | --- |
| G0 | init 在 0.5 m 内 | 解码后仍在 0.5 m 内的比例（保持率） |
| G1 | init 在 (0.5, 1] m | 解码后进入 0.5 m 的比例（**修正率**） |
| G2 | init 在 (1, 2] m | 解码后进入 0.5 m 的比例；解码后进入 1 m 的比例 |

每组给出 GT 数，按 <30 / ≥30 / 全部分列。

**另需输出**：`N_missed_D1` = 全部 GT 中 `rank_ca`（2 m 列）≥ 200 的个数，按 <30 / ≥30 分列，作为 S8 的参照。

注意：最近的同类初始 query 不一定就是产生该最终中心的那个 query，所以 B0-2 是近似量。

## 6. 第 2 步：oracle E0–E3（单卡，串行）

```bash
CONFIG=<CONFIG> CKPT=<CKPT> CFG_OPTS=<与 #1 相同> CUDA_VISIBLE_DEVICES=<空闲卡> \
  BASE_MAP=0.665463 BASE_NDS=0.706154 \
  bash tools/query_diag/run_diag.sh exec2
```

| 组 | 参数 | 目的 |
| --- | --- | --- |
| E0 base | `--oracle none` | S5'：改动后的代码不改变 base 结果 |
| E1 missed-orig | `--oracle loc --subset missed --range 0 1000 --score orig` | coverage_orig：只补漏检、分数还原 |
| E2 missed-boost | `--oracle loc --subset missed --range 0 1000 --score boost` | coverage_boost：只补漏检、保留高分数（上限） |
| E3 jitter | `--oracle loc --subset all --range 0 1000 --score orig --jitter-cells 1 --seed 0` | 增益对精确格子位置的依赖 |

预计每组约 38 分钟，共约 2.5 GPU 小时。

## 7. 自检

任一项不通过就停下回传，不得为通过而改口径。

| 自检 | 适用 | 判据 |
| --- | --- | --- |
| S0' | 第 1 步 | 见第 4 节 |
| S9 | B0 | 从 npz 重算的 K=200、@2 m class-aware 召回率与 #1 第 2 步完全相同：R_near(200) = 0.97319，R_far(200) = 0.94944（5 位小数） |
| S5' | E0 | mAP、NDS 与参考 base（0.665463 / 0.706154）相差各 ≤ 0.001 |
| S6 | E1–E3 | `init_mismatch_rate` ≤ 0.01；`capped_cells` = 0 |
| S7' | E1–E3 | `selected_injected` ≥ 0.99 × `injected_cells`；orig 模式（E1、E3）下 `restored_queries` = `selected_injected` |
| S8 | E1、E2 | `missed_gt` 与 `N_missed_D1` 的相对差 ≤ 1%；E1 与 E2 的 `missed_gt` 相对差 ≤ 1% |

## 8. 预先登记的判读规则（已确认）

**定义**：
- coverage_orig = ΔmAP(E1)，coverage_boost = ΔmAP(E2)，均相对 E0，单位 pp。
- jitter 保留率 = ΔmAP(E3) ÷ 1.875。分母取 #1 的 all 组，两次 base 之间的非确定性约 0.07 pp，对分母的影响 < 4%，可以忽略。
- 任一量距其阈值 ≤ 0.1 pp（保留率为 ≤ 0.03）时，审阅中标注"临界"；结论仍按规则下，但会说明不确定性。

| 规则 | 判据 | 结论 |
| --- | --- | --- |
| Q1 覆盖 | coverage_orig ≥ 1.0 pp | 覆盖是主要瓶颈，Dome-DETR 继续，叙事用"召回不足 / 拥挤场景" |
| | 0.5 ≤ coverage_orig < 1.0 pp | 覆盖是次要因素，Dome-DETR 不作为主方法 |
| | coverage_orig < 0.5 pp | 覆盖方向关闭，停止 Dome-DETR 线 |
| Q2 覆盖上限 | coverage_boost | 只作参考，不单独决定去留 |
| Q3 位置依赖 | 保留率 ≤ 0.3 | 增益依赖精确的初始位置，C 的前提成立 |
| | 保留率 ≥ 0.7 | C 的前提不成立，需要重新找机制 |
| | 0.3 < 保留率 < 0.7 | 审阅讨论 |

**B0 的解读参考**（只用于解释机制，不参与决策）：G1 修正率 ≤ 0.3 表示解码器基本沿用初始位置，提升空间在 heatmap 峰值或 query 放置；≥ 0.6 表示解码器已能修正大部分偏差，oracle 增益需要另找原因；介于两者之间为混合。

**判读顺序**：

1. S0'、S5'–S9 全部通过？否 → 不下结论，排查后重跑。
2. Q1 ≥ 1.0 → Dome-DETR 继续（召回不足叙事）；若 Q3 ≤ 0.3，C 可作为第二个贡献点。
3. Q1 在 0.5–1.0 → Dome-DETR 不作为主方法；Q3 ≤ 0.3 时以 C 为主线，覆盖可作为其中一个组件；其他情况审阅讨论。
4. Q1 < 0.5 → 停止 Dome-DETR 线；Q3 ≤ 0.3 时转向 C，执行文档 #1、#2 的结果作为动机分析；Q3 ≥ 0.7 时两个方向都不成立，重新审视机制；介于之间审阅讨论。

**事先记录的预测**（不影响判读，只用于事后对照）：coverage_orig < 0.5 pp；jitter 保留率 ≤ 0.3。

## 9. 回传材料清单

每一步结束后立即回传。

- [ ] M1–M5 的 diff 摘要；门禁输出；S0' 结果
- [ ] B0：txt 与 json 全文；S9 结果；`N_missed_D1`
- [ ] E0：`oracle_*.json`、S5' 结果（E0 通过后再继续，但不必等审阅）
- [ ] E1–E3：各组 `oracle_*.json`；汇总输出（ΔmAP / ΔNDS、保留率、S6–S8）；`oracle_breakdown.py` 输出（各匹配阈值的 AP 贡献、分类别 ΔAP、mATE）
- [ ] 每组的 `runtime` 字段和日志文件路径

## 10. 禁止事项

- 修改 `projects/BEVFusion` 下的源码、配置或权重。
- 改变"漏检"定义、`--miss-thr`、jitter 幅度、种子，或第 8 节的任何阈值。
- 增跑 E0–E3 之外的 oracle 变体（需先申请）。
- S5' 不通过时继续跑 E1–E3。
- 运行任何训练；使用超过 1 张 GPU；在他人占用的卡上运行。
- 删除或覆盖 `work_dirs/diag/` 下 #1 的结果。

## 11. 已知局限

- **E1 会略微高估纯覆盖收益**：被注入的漏检目标同时拿到了真值所在格子的位置，混入了一部分定位收益。这个偏差的方向对"停止"结论是保守的。
- **`loc` 模式下类别由网络决定**：执行文档 #1 显示远距离漏检中约一半属于类别锁定，这部分在 E1 中修不回来。这是有意的设计，因为只改 query 分配的方法同样改不了类别。
- **E2 泄漏了"这里有目标"的置信度**，只能作为上限。
- **E3 只用一个随机种子**：约 11.8 万个 GT 平均后的波动很小，但不是零。
- **B0-2 是近似量**：见第 5 节的说明。
- 结论只针对这一个权重；该权重比 BEVFusion 原论文 val 结果低约 2 pp。
