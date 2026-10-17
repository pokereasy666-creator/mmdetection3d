# 执行文档 #1（v1.1）：TransFusion 头 query 初始化瓶颈诊断

- 版本：v1.1，2026-10-17（取代 v1）
- 执行者：Claude Code（离线服务器）
- 结果审阅：Claude（对话侧），按第 10 节「预先登记的判读规则」
- 决策者：Hermanuh
- 当前状态：第 0 步源码核对已通过（`EXEC01_STEP0_SOURCE_AUDIT.md`，A1–A9 一致）；下一步是第 5 节「服务器一致性检查」

## 0. 相对 v1 的变更

| 编号 | 变更 | 来源 |
| --- | --- | --- |
| V1 | 批准 C1–C6，其中 C2、C4、C5 有补充要求（第 4 节） | 第 0 步报告 F1–F7 |
| V2 | 新增第 5 节「服务器一致性检查」S0a–S0d，在第 1 步之前执行 | 审阅新增 |
| V3 | 新增自检 S4a（center 快照必须是格子单位） | 审阅新增，配合 C1 |
| V4 | S5 不通过时，按预登记流程用无 hook 的 `tools/test.py` 裁决 | 审阅新增，配合 C2 |
| V5 | CONFIG 固定为 4xa30 配置；部署到服务器现有目录 | 第 0 步报告第 5 节 |
| V6 | 局限中补充"该权重低于原论文 val 约 2 pp" | 审阅新增 |

## 1. 文档信息与输入

本文档交给 Claude Code 执行：不训练，判断 BEVFusion 的 TransFusion 头在 query 初始化环节是否存在足够大的提升空间，以决定 Dome-DETR 迁移方向是否立项。

| 项 | 内容 |
| --- | --- |
| 部署位置 | 服务器现有目录 `mmdetection3d-claude-jolly-wozniak-c4YMQ`；只把 `tools/query_diag/` 拷入，前提是 S0a 通过 |
| CONFIG | `projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_4xa30-amp-accum_nus-3d.py` |
| CKPT | `<用户填写：baseline1/epoch_19.pth 的服务器绝对路径>` |
| BASE_MAP / BASE_NDS | `<用户填写：epoch_19 验证日志中的 4 位小数值>` |
| R1 阈值 | `<用户填写：1.0 或 0.8 pp>`，第 3 步开跑前必须填好 |
| 数据 | nuScenes val，6019 帧 |

## 2. 目标与待验证假设

待验证假设 H0：TransFusion 头在初始化阶段漏掉了相当比例的远距离目标（heatmap top-K 偏向近处强响应），并且这部分漏检对 mAP 的影响足够大。

| 编号 | 回答的问题 | 方法 | 对应决策 |
| --- | --- | --- | --- |
| D1 | 每个 GT 附近有没有初始 query？按距离、类别、预算 K 统计 | 复现 head 初始化规则，统计初始 query 召回率和解码后中心召回率 | 漏检是否集中在远处；加大或重分配预算能否找回 |
| D2 | 初始化位置全对时，mAP/NDS 最多能涨多少 | 测试时把 GT 位置注入 heatmap，预算不变，不泄漏 GT 置信度 | 只改 query 分配/初始化的方法的上限 |
| D3 | 每帧 GT 数是否会超过 num_proposals（200） | 统计每帧 GT 数分布 | "可变 query 数量"是精度贡献还是效率贡献 |

## 3. 环境与硬约束

与本文档其他部分冲突时，以本节为准。

- 代码基：mmdet3d 1.4.0 + `projects/BEVFusion`，源码模式，所有命令在仓库根目录执行，`PYTHONPATH=$PWD`。服务器 Python 3.8。
- 算力：共享 4×A30。D1 用 1 卡；D2 默认 1 卡，最多 2 卡。目标卡上有任何他人进程或显存占用 > 1 GiB 时不得启动（C5 自动检查）。
- 不改 baseline：不修改 `projects/BEVFusion` 下任何源码、配置和权重。所有诊断代码只放在 `tools/query_diag/`。
- 同步门禁：所有 .py 文件无非 ASCII 字符，`ast.parse(..., feature_version=(3, 8))` 通过；`run_diag.sh` 过 `bash -n`。
- 不训练：只做推理和评测。

## 4. 参考实现的修改（批准清单）

以下改动全部限定在 `tools/query_diag/` 内。实施后回传 diff 摘要。

| 编号 | 改动 | 审阅补充要求 |
| --- | --- | --- |
| C1 | 在替换后的 `forward` 中、`predict_by_feat` 之前，存 `STATE['center_grid'] = center[..., -P:].detach().clone()`；T4 改读这份快照 | 加自检 S4a（第 6 节） |
| C2 | `oracle` 入口先只跑 base，用 `BASE_MAP` / `BASE_NDS` 自动判 S5（容差各 0.001）；未设置或不通过即停，不跑后四组 | S5 不通过时执行第 8 节的裁决流程，不得直接调整已知值 |
| C3 | `CONFIG` 改为必填；新增可选 `CFG_OPTS` 透传给 `--cfg-options`，默认空 | `CFG_OPTS` 非空时，必须写入输出 json 并在回传中说明 |
| C4 | 新增 `diag_merge_bins.py` 和 `merge` 入口，从 `recall_per_gt.npz` 计算 "<30 m" 与 "≥30 m" 合并段 | 输出第 10 节所需的全部规则输入量（见第 7 节清单）；只给 S 类自检的 PASS/FAIL，**不输出 R 类规则的判定** |
| C5 | 启动前检查目标卡 | 改为必选，不提供跳过开关。判据：有计算进程，或 `memory.used` > 1024 MiB（跨容器时进程列表可能看不到他人进程） |
| C6 | json 中追加 `runtime`：总耗时、峰值显存（allocated / reserved）、torch / mmcv / mmengine / mmdet3d 版本、仓库 VERSION | 另记录 CONFIG 和 CKPT 的 md5，以及测试循环的 fp16 设置 |

## 5. 服务器一致性检查（S0，第 1 步之前）

第 0 步核对的是 `82c2b4e`，服务器工作目录不一定处于这个状态。下列任一项不通过就停下报告。

| 自检 | 判据 |
| --- | --- |
| S0a | 服务器上 `git diff --stat 82c2b4e -- projects/BEVFusion` 为空，且没有未提交改动；若服务器不是 git 工作区，则 `transfusion_head.py`、`utils.py`、`bevfusion.py`、`sparse_encoder.py` 和三份相关配置的 md5 与审计版本逐一相同 |
| S0b | 列出 4xa30 配置相对官方融合配置的全部覆盖项；`model` 部分不得引入任何非官方模块类型（例如 EP-Fusion 的模块） |
| S0c | 加载 CKPT 的日志中没有 `missing keys`、`unexpected keys`、`size mismatch` |
| S0d | 确认 `test_dataloader` 与 `val_dataloader` 指向同一个 val 标注文件，`test_evaluator` 与 `val_evaluator` 设置相同；记录 `val_cfg` / `test_cfg` 的 fp16 设置；BASE_MAP / BASE_NDS 必须来自与本次诊断相同的评测设置 |

## 6. 第 1 步：部署、门禁与 300 帧冒烟

1. 实施 C1–C6，跑同步门禁。
2. 冒烟（单卡）：

   ```bash
   CONFIG=<CONFIG> CKPT=<CKPT> CUDA_VISIBLE_DEVICES=<空闲卡> \
     bash tools/query_diag/run_diag.sh quick
   bash tools/query_diag/run_diag.sh merge   # 对冒烟结果跑一次，验证 C4
   ```

3. 记录耗时和峰值显存，估算全量时间。

S1–S4a 全部通过才能进入第 2 步。任一项失败就停下回传，不得为了通过而改统计口径。

| 自检 | 判据 | 失败指向 |
| --- | --- | --- |
| S1 | 初始化复现与 `head.query_labels` 匹配率 ≥ 0.99 | A1 |
| S2 | 日志打印 `bev_pos transposed=False` | A3 |
| S3 | 无 `[WARNING]`；0–20 m 段 class-aware 初始召回率 @2 m ≥ 0.5 | A3、A7 |
| S4a | `center_grid` 快照中 ≥ 99% 的值落在 [−2, 182]（格子单位） | C1 未生效 |
| S4 | T4 中 0–20 m 段 `final_agn` ≥ `init_agn` − 0.05 | 解码中心换算 |

## 7. 第 2 步：全量 D1 + D3

```bash
CONFIG=<CONFIG> CKPT=<CKPT> CUDA_VISIBLE_DEVICES=<空闲卡> \
  bash tools/query_diag/run_diag.sh recall
bash tools/query_diag/run_diag.sh merge
```

固定参数（不得为结果调整）：K = 200 / 300 / 500 / 1000；中心距离阈值 0.5 / 1 / 2 / 4 m，主阈值 2 m；分段 0–20 / 20–30 / 30–40 / 40–50 m；开启 nuScenes 按类别评测距离截断。

`merge` 必须输出的规则输入量（全部为 class-aware @2 m，除非注明）：

| 量 | 定义 |
| --- | --- |
| R_near(K)、R_far(K) | "<30 m"、"≥30 m" 段在 K = 200 / 1000 下的初始召回率 |
| miss_near、miss_far、比值 | miss = 1 − R(200)；比值 = miss_far / miss_near |
| ΔR_far | R_far(1000) − R_far(200) |
| agn_gap_far | "≥30 m" 段 class-agnostic R(200) − class-aware R(200) |
| final_gap_far | "≥30 m" 段 `final_ca` − R_far(200) |
| frac_over_200 | 每帧有效 GT 数 > 200 的帧占比 |
| 各段 GT 数 | 两个合并段的 GT 数量，用于判断统计量是否足够 |

## 8. 第 3 步：D2 oracle 五组

```bash
CONFIG=<CONFIG> CKPT=<CKPT> BASE_MAP=<4 位小数> BASE_NDS=<4 位小数> GPUS=1 \
  bash tools/query_diag/run_diag.sh oracle
bash tools/query_diag/run_diag.sh summary
```

建议 `GPUS=1`，与 base 保持同一执行路径（DDP 路径未经测试）。

| 组 | 参数 | 回答的问题 |
| --- | --- | --- |
| base | `--oracle none` | 挂上 hook 后能否复现原始结果 |
| all | `--oracle loc --range 0 1000` | 全部 GT 位置初始化正确时的上限（类别仍由网络决定） |
| far | `--oracle loc --range 30 1000` | 只修正 ≥30 m 目标时的增益 |
| near | `--oracle loc --range 0 30` | 只修正 <30 m 目标时的增益 |
| all_cls | `--oracle loc_cls --range 0 1000` | 位置和类别都给真值时的上限 |

所有组 `--score orig`。

| 自检 | 判据 |
| --- | --- |
| S5 | base 的 mAP、NDS 与 BASE_MAP / BASE_NDS 相差各 ≤ 0.001 |
| S6 | 四个 oracle 组 `init_mismatch_rate` ≤ 0.01，`capped_cells` = 0（不为 0 时如实报告） |
| S7 | 四个 oracle 组 `restored_queries` ≥ 0.99 × `injected_cells` |

**S5 不通过时的裁决流程（预先登记）**：

1. 停止，不跑后四组。
2. 用同一 CONFIG / CKPT 跑一次无 hook 的 `tools/test.py`，单卡。
3. 若 hook 版 base 与无 hook 版的 mAP、NDS 相差各 ≤ 0.0005：问题在已知值来源而不在 hook。以无 hook 版结果作为新的参考值，回传两组数值和差异原因的推测，**等确认后**再继续。
4. 否则：hook 改变了推理结果，停止并回传。

## 9. 回传材料清单

每一步结束后立即回传，不要攒到最后。npz 不必回传。

- [ ] S0a–S0d 逐项结果（含 S0b 的覆盖项列表）
- [ ] C1–C6 的 diff 摘要；门禁输出
- [ ] 第 1 步：冒烟 `recall_report.txt`、冒烟 `merge` 输出、S1–S4a 逐项结果、耗时和峰值显存
- [ ] 第 2 步：全量 `recall_report.txt`、`recall_summary.json`、`merge` 输出（第 7 节全部量）
- [ ] 第 3 步：5 个 `oracle_*.json`、`summary` 输出、S5–S7 逐项结果；若触发 S5 裁决，附无 hook 版结果
- [ ] 每份 json 的 `runtime` 字段

## 10. 预先登记的判读规则

阈值在第 3 步开跑前确定，之后不得因结果修改。审阅按下列顺序进行，前一步不通过即停在该步结论。

定义：ΔmAP_X = X 组 mAP − base 组 mAP（pp）。其余量见第 7 节。

| 规则 | 判据 | 含义 |
| --- | --- | --- |
| R1 总上限 | ΔmAP_all ≥ `<R1 阈值>` pp | 不满足：停止 Dome-DETR 主线 |
| R2 远距离特异性 | ΔmAP_far ≥ 0.5 pp，且 miss_far / miss_near ≥ 2 | 不满足：方法可做，但远距离不作为主卖点 |
| R3 预算能否找回 | ΔR_far ≥ 0.05 | ≤ 0.02：漏检源于 heatmap 本身，转向增强特征 / heatmap；0.02–0.05：审阅讨论 |
| R4 类别锁定 | ΔmAP_all_cls − ΔmAP_all ≥ 1.0 pp，或 agn_gap_far ≥ 0.05 | 满足：瓶颈在初始化类别判错，转向类别解耦的初始化 |
| R5 解码器补救 | final_gap_far ≥ 0.05 | 满足：方法设计时下调对初始化改进的预期 |
| R6 可变数量 | frac_over_200 < 0.5% | 满足："可变 query 数量"只能写成效率贡献 |

判读顺序：

1. S0–S7 全部通过？否 → 不下结论，排查后重跑。
2. R1 满足？否 → 停止 Dome-DETR 主线。
3. R4 触发？是 → 转向类别解耦的初始化。
4. R3：ΔR_far ≥ 0.05？否 → ≤ 0.02 转向增强特征 / heatmap；0.02–0.05 审阅讨论。
5. R2 满足？是 → Go，以远距离漏检为主线；否 → Go，但叙事改为召回不足 / 拥挤场景。

R5、R6 不影响是否立项，只影响方法设计。

## 11. 禁止事项

- 修改 `projects/BEVFusion` 下的源码、配置或权重。
- 看到结果后调整第 10 节阈值或任何统计口径；为通过自检而改口径。
- 运行任何训练。
- 使用超过 2 张 GPU，或在他人占用的卡上运行。
- 删除或覆盖已有 `work_dirs`；输出只写到 `work_dirs/diag/`。
- S0 未通过就部署运行；跳过 base 组直接跑 oracle 组。
- S5 裁决流程第 3 步未经确认就更换参考值继续。

## 12. 已知局限与风险

- nuScenes 按类别截断评测距离：锥桶、路障只到 30 m，行人和两轮车只到 40 m。"≥30 m" 段以车辆类为主；远处小锥桶不计入官方 mAP。
- 自车距离用 LiDAR 坐标系 xy 范数近似，偏差约 1 m 量级。
- oracle 上限偏保守：注入的 query 沿用原始 heatmap 分数，只测"修正初始化位置"的收益。
- D2 为负只能排除"只改 query 分配/初始化"的方法，不能排除增强特征或 heatmap 的方法。
- 本次权重 val mAP / NDS 约 0.665 / 0.696，低于 BEVFusion 原论文 val 的 68.5 / 71.4 约 2 pp。较弱 baseline 上测到的初始化空间可能偏大，结论不能直接外推到更强的 baseline。
- 所有结论只针对这一个权重。
