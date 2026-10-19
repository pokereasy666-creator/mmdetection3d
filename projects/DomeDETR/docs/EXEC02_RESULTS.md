# 执行文档 #2 · 结果记录

- 执行文档：`exec_doc_02_coverage_vs_localization.md`；实施说明：`EXEC02_IMPLEMENTATION.md`；决策：`EXEC02_DECISIONS.md`
- 工作区：`/212022085500129/dome-detr/mmdetection3d-claude-dome-detr-query-diag`（A-1）
- 本文件只记录结果与自检。Q 类规则只适用于 E1–E3，判读留给审阅

## 第 1 步 S0' 与 B0（2026-10-19 回传）：S0' PASS，S9 PASS

原始数据都在 `results/exec02/` 下：

- `step1_s0p_check.txt`
- `step1_b0_decoder.txt`、`step1_b0_decoder.json`
- `step1_b0_terminal.txt`（终端全文，即服务器上的 `work_dirs/diag/logs/b0_20261019_060638.log`；改名是因为 .gitignore 忽略 *.log）

**S0'**

- CONFIG md5 `8f49ac79…`、CKPT md5 `c16536e3…`、CFG_OPTS 均与 #1 相同。
- `tools/query_diag/` 中有改动的正好是登记的 7 个文件：M1–M5，以及按 E2-1 / E2-2 允许的 LOG 改动和迁移。md5 与 `EXEC02_IMPLEMENTATION.md` 第 5 节一致，没有 OTHER 类改动。
- 门禁全部通过。

**S9**：R_near(200) = 0.97319，R_far(200) = 0.94944，5 位小数与 #1 第 2 步完全相同。

**开发侧复核**（用回传的 json 计算）：

- B0-1 各分段、各阈值的 init 列与 #1 第 2 步 `recall_summary.json` 的 T1（class-aware）逐项相同。
- 每个分段内，G0 + G1 + G2 + 漏检 = GT 总数（全部 118538，<30 m 86121，≥30 m 32417）。
- N_missed_D1 = N_gt × (1 − R(200))：全部 3948，<30 m 2309，≥30 m 1639。

**S8 的参照区间**：E1、E2 的 `missed_gt` 与 3948 相差 ≤ 1%，即落在 3909–3987 之间。

### B0 的要点（描述性，不做 Q 判定）

| 量 | <30 m | ≥30 m | 全部 |
| --- | --- | --- | --- |
| 0.5 m 召回 init → final | 0.871 → 0.905（+3.42 pp） | 0.713 → 0.767（+5.40 pp） | 0.828 → 0.868（+3.96 pp） |
| 2 m 召回 init → final | 0.9732 → 0.9735 | 0.9494 → 0.9497 | 0.9667 → 0.9670 |
| G0（init ≤ 0.5 m）保持率 | 0.991 | 0.967 | 0.985 |
| G1（0.5–1 m）进入 0.5 m 的比例 | 0.544 | 0.440 | 0.497 |
| G2（1–2 m）进入 0.5 m / 1 m 的比例 | 0.000 / 0.188 | 0.003 / 0.162 | 0.002 / 0.175 |
| final 在 0.5 m 内没有同类中心的 GT 占比 | 9.5% | 23.3% | 13.2% |

1. **解码器只修正定位，不改变覆盖。** 0.5 m 召回提升 4 pp，40–50 m 段达 +7.1 pp；2 m 和 4 m 召回的变化都不超过 0.11 pp（最大为 40–50 m 段 4 m 的 −0.10 pp），与 #1 的 R5 一致。
2. **修正范围大约是一个 BEV 格子。** 格子边长为 0.075 m × 8 = 0.6 m。初始位置偏 0.5–1 m 的 GT，约一半能被修到 0.5 m 内；偏 1–2 m 的几乎修不到 0.5 m 内（0.2%），只有约 18% 能进入 1 m。
3. **解码器很少把好的位置改坏。** G0 保持率为 98.5%，远距离为 96.7%。
4. **0.5 m 级定位的余量远大于覆盖的余量。** 解码后，13.2% 的 GT（≥30 m 段为 23.3%）在 0.5 m 内没有同类中心；而 2 m 内没有初始 query 的只有 3.3%。#1 oracle 的增益几乎全部出现在 AP@0.5 m，与这个余量所在的位置一致。
5. **与 E3 的关系**（只说明 E3 在测什么，不作预测）：
   - jitter 在两个方向上各自从 {−1, 0, +1} 格中随机取值，因此 1/9 的注入落在原格，4/9 平移 0.6 m，4/9 斜移约 0.85 m。
   - 这些偏移大致落在 B0-2 的 G1 / G2 距离范围内。
   - E3 会直接测出这种偏移下增益还剩多少。

局限：

- B0-2 是近似量：最近的同类初始 query 不一定就是产生该最终中心的那个 query。
- final 召回统计的是"任一同类最终中心落在阈值内"，不是与评测匹配的那个框。

## 第 2 步 E0–E3：待执行

代码与第 1 步相同，不需要重新部署。

```bash
cd /212022085500129/dome-detr/mmdetection3d-claude-dome-detr-query-diag
source tools/query_diag/exec02_env.sh
nvidia-smi
CUDA_VISIBLE_DEVICES=<空闲卡号> bash tools/query_diag/run_diag.sh exec2
```
