# 执行文档 #2 · 决策记录

- 登记日期：2026-10-19，在会话中经用户确认
- 效力：在第 1 步（`b0`）开跑前登记。登记之后，不得因任何结果修改下列范围和取值
- 适用：`exec_doc_02_coverage_vs_localization.md`；执行文档 #1 的决策（`EXEC01_DECISIONS.md` 中的 D-1 至 D-4、A-1）继续有效

| 编号 | 决策 | 取值 |
| --- | --- | --- |
| E2-1 | S0' 允许范围 | 用户原文："S0' 允许范围：M1–M5，以及 #1 收尾时登记的日志持久化改动（`diag_summarize.py`、`run_diag.sh` 的日志部分）。条件：日志改动不改变任何计算结果，以 #1 数据重跑输出逐字节一致为证。"<br>理由：S0' 写"只允许 M1–M5"，本意是排除会影响推理结果的改动。这两处改动只读 json 或只转存终端输出，不参与推理；日志持久化也是 #2 第 2 节明确要求的。<br>落实：`diag_exec2_check.py s0p` 把这两个文件归为 "LOG … decision E2-1"；`tools/query_diag/` 中其他任何未登记的改动都归为 OTHER，S0' 判 FAIL。<br>证据：`results/exec02/dev_e2-1_log_equivalence.txt`，由 `tests/test_exec2_cpu.py --evidence` 生成，结果见下方 |
| E2-2 | `oracle_breakdown.py` 迁移 | 计入 M5。它从 `projects/DomeDETR/tools/` 移到 `tools/query_diag/`，随诊断目录部署。用 #1 的 5 个 json 重新生成的输出与归档的 `results/exec01/step3_oracle_breakdown.txt` 逐字节相同；run 列宽度自适应只改变显示，#1 的 tag 最长 7 个字符，列宽仍为 8 |
| E2-3 | M1 漏检判定的 K | 读取 head 的属性：`diag_oracle_query.py:103` 为 `P = int(head.num_proposals)`，与 D1 的 `diag_query_recall.py:196` 写法相同；漏检条件为 `rank_ca >= P`。B0 的 `N_missed_D1` 用 npz 中由 D1 记录的 `P`。代码中没有写死 200。<br>证据：mock 的 num_proposals 设为 40 时，`missed_gt` = 237，等于 D1 的 `rank_ca >= 40` 计数；如果写死 200，会得到 96 |

## E2-1 证据

- **`diag_summarize.py`**：#1 版本（`484b6bb`，md5 `a86cf0a3`）与当前版本（md5 `deaa94b3`）对比。
  - 输入：#1 回传的 5 个 `oracle_*.json`、`s5_gate.json` 和 `step2_recall_summary.json`，按服务器上的相对路径 `work_dirs/diag/...` 摆放。
  - #1 代码的终端输出与 `step3_summary_reconstructed.txt` 逐字节相同（2676 字节）。
  - 当前代码写出的 `summary.txt` 与归档逐字节相同。
  - 当前代码的终端输出 = 归档全文 + 最后一行 `[summary] saved work_dirs/diag/summary.txt`。这一行是写文件动作本身的提示。
  - 文件层面：只多出 `summary.txt`，输入文件全部未变。
- **`run_diag.sh`**：#1 版本（`de1bbcd`，md5 `e19bf9b4`）与当前版本对比。#1 一侧使用当时的整个 `tools/query_diag/`，其 md5 集合与 #1 的 runtime 记录完全一致。
  - 覆盖 #1 的全部入口共 13 个场景，包括 build-ops、s0、quick、recall、单卡 / 双卡 oracle、S5 不通过、s5-adjudicate、merge、summary，以及三种拒绝情形。
  - 以下各项全部相同：退出码、每一条外部调用的 argv、所有产出文件的 md5（只屏蔽 gate / merge json 中的墙钟 `time` 字段）。
  - 终端输出的差别只有开头一行 `[run_diag] step=<入口> log=<路径>`，日志文件内容与终端输出相同。
  - 文件层面：只多出 `logs/<入口>_<时间>.log`（summary 入口另多出上面的 `summary.txt`）。
  - 敏感性：把 `--max-samples 300` 改成 301 后，对比测试即判 FAIL。
- **口径说明**：两处日志改动都只在终端多打印一行提示，`run_diag.sh` 是开头的日志路径，`diag_summarize.py` 是末尾的保存提示。除这两行外，输出逐字节一致，计算结果和产出文件没有变化。
