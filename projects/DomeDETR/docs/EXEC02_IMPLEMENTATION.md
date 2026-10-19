# 执行文档 #2 实施说明（M1–M5）

- 执行文档：`exec_doc_02_coverage_vs_localization.md`（本目录）
- 分支：`claude/dome-detr-query-diag`
- 状态：代码与开发侧测试完成；决策 E2-1 至 E2-3 已登记（`EXEC02_DECISIONS.md`）；第 1 步 S0'、S9 PASS，第 2 步 E0–E3 已完成且 S5'–S8 PASS（`EXEC02_RESULTS.md`）；E3 的有效性问题与 E3' 申请待审阅
- 改动范围：只在 `tools/query_diag/` 内（另有开发侧测试 `projects/DomeDETR/tests/`，不部署）。`projects/BEVFusion` 未改动

## 1. diff 摘要

| 编号 | 文件 | 改动 |
| --- | --- | --- |
| M1 | `diag_oracle_query.py` | 新增 `--subset {all,missed}`（默认 `all`）、`--miss-thr`（默认 2.0）。在 heatmap hook 内、**修改 heatmap 之前**，用 `replicate_init(head, logits, P)` 取未修改 heatmap 的 K = num_proposals 个初始 query（`P = int(head.num_proposals)`，与 D1 同一写法，未写死 200，见 E2-3）；对每帧先用 `gt_from_sample` 取 GT（按类别评测距离截断），再按 `--range` 过滤，然后调用 `per_gt_hits(..., [miss_thr])` 的 class-aware 结果，`rank_ca >= P` 即为漏检。`missed` 模式只把漏检 GT 交给 `oracle_cells`，注入方式与 `loc` 相同（通道 = 该格原始 logits 的 argmax）。三个函数都与 D1 共用（`diag_common.py` 未改动） |
| M2 | `diag_oracle_query.py` | 计数器由 6 个增加到 9 个：`selected_injected`（被选中的初始 query 中 (类别, 格子) 落在注入掩码上的个数，orig 与 boost 都统计）、`gt_in_range`、`missed_gt`。另外：`[oracle]` 打印行增加 subset / miss_thr / seed；`selected_injected < 0.99 × injected_cells` 时打印 WARNING |
| M3 | `diag_b0_decoder.py`（新） | 读取 #1 的 `recall_per_gt.npz`，输出表 B0-1、B0-2、`N_missed_D1` 和 S9，写入 `b0_decoder.txt` / `.json`。S9 不通过时退出码为 1 |
| M4 | `run_diag.sh` | 新增入口 `b0`、`exec2`、`exec2-summary`（详见第 2.4 节）；`run_oracle` 支持 `ORACLE_ROOT`，exec2 的输出因此写到 `work_dirs/diag/exec2/` |
| M4 | `exec02_env.sh`（新） | `source exec01_env.sh`（CONFIG / CKPT / CFG_OPTS 与 #1 完全相同），再设置 `BASE_MAP=0.665463`、`BASE_NDS=0.706154`（本文档第 1 节的参考 base） |
| M5 | `diag_exec2_check.py`（新） | 子命令 `s0p`（S0'）、`step1-ok`（S0' 与 S9 都通过才返回 0）、`summary`（M5 汇总） |
| M5（E2-2） | `oracle_breakdown.py` | 从 `projects/DomeDETR/tools/` 移到 `tools/query_diag/`（随诊断目录一起部署）。逻辑不变，只抽出 `breakdown_lines()` 供汇总调用，并让 run 列宽度随 tag 长度自适应。对 #1 的 5 个 json 输出与归档的 `results/exec01/step3_oracle_breakdown.txt` **逐字节相同** |

"旧模式行为与 #1 完全相同"的证明：开发侧测试从提交 `de1bbcd` 取出 #1 实际运行的 `diag_oracle_query.py` / `diag_common.py`（md5 与 #1 服务器 runtime 记录的 `22880404…` / `603244dc…` 一致），在同一个 mock 上分别跑 `--oracle none`、`--oracle loc --range 0 1000`、jitter 变体，新旧两版的 metrics 与 #1 已有的全部 stats 字段完全相等。

## 2. 关键实现细节

### 2.1 漏检判定（M1）

- 判定在每次 forward 内、逐帧完成，使用的 heatmap 是注入之前的原始 logits。
- 漏检计数对所有注入模式都会计算（`all` 模式同样记录 `gt_in_range` / `missed_gt`）。只有 `--subset missed` 时才据此筛选注入对象。
- `--miss-thr` 只出现在命令行；`exec2` 不传这个参数，因此固定使用默认值 2.0。

### 2.2 计数（M2）

`selected_injected` 在 head forward 之后统计，此时初始 query 的 (类别, 格子) 已确定：`oracle_mask.gather(1, cls·H·W + idx)` 求和。它在分数还原之前计算，与 `--score` 无关。

### 2.3 B0（M3）

- init@t = `rank_ca[:, t] < P`；final@t = `fin_ca[:, t]`。P 取 npz 内记录的值（200）。
- 分段：0–20、20–30、30–40、40–50、<30、≥30、全部。<30 / ≥30 的定义与 #1 的 `diag_merge_bins.py` 相同（`dist < 30` / `dist >= 30`）。
- B0-2 的分组：
  - G0 = init@0.5
  - G1 = ¬init@0.5 ∧ init@1
  - G2 = ¬init@1 ∧ init@2
- B0-2 的统计量：
  - G0 的保持率 final@0.5
  - G1 的修正率 final@0.5
  - G2 的 final@0.5 与 final@1
- N_missed_D1 = ¬init@2 的计数，按 <30 / ≥30 / 全部分列。
- S9 用 `'%.5f'` 格式化后做字符串比较，要求严格等于 0.97319 / 0.94944。#1 的 `recall_merged.txt` 用同一份 npz、同一定义得到这两个值。

### 2.4 `run_diag.sh` 入口（M4）

**`b0`（不占 GPU）**

- 前置条件：`work_dirs/diag/oracle_base/oracle_base.json`（#1 第 3 步）和 `work_dirs/diag/recall/recall_per_gt.npz`（#1 第 2 步）都存在。
- 先运行 `diag_exec2_check.py s0p`。不通过则退出（退出码 4），不运行 B0。
- 再运行 `diag_b0_decoder.py`。S9 不通过则退出（退出码 4）。

**`exec2`（以下检查依次进行，任一不满足就拒绝启动）**

1. `BASE_MAP` / `BASE_NDS` 必须等于 0.665463 / 0.706154，否则拒绝启动，并提示 `source exec02_env.sh`。这一步防止误用 #1 的训练日志参考值 0.6648 / 0.7060。
2. `step1-ok`：第 1 步（S0' 与 S9）未通过则拒绝启动。
3. 强制 `GPUS=1`，单卡运行，不用 torchrun。
4. 每次 `run_oracle` 启动前都会执行 C5 GPU 检查（共 4 次）。若 `CUDA_VISIBLE_DEVICES` 指定的卡不是恰好 1 张，或该卡已有计算进程或显存占用 > 1 GiB，则拒绝。

通过后的运行顺序：

1. 跑 E0。
2. 用 `diag_gate_s5.py gate` 判定 S5'，结果写入 `exec2/oracle_e0_base/s5p_gate.json`。不通过就停（退出码 4），E1–E3 不会运行。终端会注明本文档没有裁决步骤。
3. 依次跑 E1、E2、E3，参数与本文档第 6 节逐字一致。
4. 生成汇总。

**`exec2-summary`**：只重新生成汇总，不跑推理。

**日志**：所有入口的终端输出都写入 `work_dirs/diag/logs/<入口>_<时间>.log`。#1 的结果目录不会被写入。

### 2.5 汇总（M5）

`exec2/exec2_summary.txt` / `.json` 的内容：

- 每组的 mAP / NDS，以及相对 E0 的 ΔmAP / ΔNDS（单位 pp）。
- coverage_orig（E1）、coverage_boost（E2）。
- E3 保留率 = ΔmAP(E3) / 1.875。
- 自检 S0'、S9、S5'、S6、S7'、S8 的 PASS/FAIL，以及每项所用的数值。缺字段时记为 MISSING。
- 以 E0 为 base 的 `oracle_breakdown` 全文：各匹配阈值的 AP 贡献、分类别 ΔAP、mATE 等 TP 误差。

**不计算 Q 类规则判定**，汇总中只有一行说明这一点。

## 3. 门禁输出

```
tools/query_diag/diag_oracle_query.py                ascii OK | ast(3.8) OK
tools/query_diag/diag_b0_decoder.py                  ascii OK | ast(3.8) OK
tools/query_diag/diag_exec2_check.py                 ascii OK | ast(3.8) OK
tools/query_diag/oracle_breakdown.py                 ascii OK | ast(3.8) OK
tools/query_diag/run_diag.sh                         ascii OK | bash -n OK
tools/query_diag/exec02_env.sh                       ascii OK | bash -n OK
tools/query_diag/diag_summarize.py                   ascii OK | ast(3.8) OK
GATES PASS
```

另外，以上 Python 文件 flake8 无告警。`diag_s0_check.py` 有 2 处缩进风格告警，它是 #1 的文件，故意不动，以保持 S0' 的 md5 一致。服务器上的 S0' 会对 `tools/query_diag/` 全部 `.py/.sh/.json` 再跑一次同样的门禁。

## 4. 开发侧测试

运行方式：CPU，Python 3.13，torch CPU 版，mmengine 0.10.5。TransFusion 的 query 初始化、`predict_by_feat` 和 decode 都是逐字复制的 mock，与 #1 的测试相同。

- `test_exec2_cpu.py`：11/11 PASS
  - `--subset all` 与 #1 代码完全相同（none / loc / jitter 三种）
  - 同一组 mock 帧上：
    - D1 的 N_missed = 96，E1 与 E2 的 `missed_gt` = 96
    - E1 注入 96 格，`all` 模式注入 398 格
    - E1 中 selected = injected = restored
    - E2（boost）中 restored = 0，被选中 query 的分数保持高值
  - jitter 在固定种子下可复现
  - B0 表与直接计算一致；S9 能正确判 PASS / FAIL
  - S0'：使用 #1 真实 md5 时只列出 M1–M5 / LOG 文件；CFG_OPTS 不同、或有未登记的文件改动时判 FAIL
  - 汇总中 Δ、保留率、S 项和 breakdown 正确，且不含 Q 判定；`N_missed_D1` 相差 3.7% 时 S8 判 FAIL
  - `run_diag.sh`：
    - 未跑 b0 时拒绝 `exec2`
    - BASE 参考值不对时拒绝
    - 两张卡时拒绝
    - S5' 不通过时只跑了 E0
    - 正常流程按 E0 → gate → E1 → E2 → E3 → 汇总 顺序执行，参数逐字核对
  - `exec02_env.sh` 的取值正确，并会拒绝在 EP-Fusion 仓库内运行
  - E2-1：`diag_summarize.py` 和 `run_diag.sh` 的日志改动用 #1 数据和 #1 全部入口重跑，与 #1 代码逐项对比。结果见 `EXEC02_DECISIONS.md`，原始输出见 `results/exec02/dev_e2-1_log_equivalence.txt`
  - E2-3：mock 的 num_proposals 设为 40 时，`missed_gt` 等于 D1 的 `rank_ca >= 40` 计数（237），不是 `>= 200` 的 96
- `test_query_diag_cpu.py`（#1 回归，带 `--ref-dir`）：18/18 PASS

## 5. S0' 开发侧预演

使用 #1 第 3 步的 `oracle_base.json`（服务器真实 runtime）作为参照：

- `CONFIG md5 8f49ac799dc93ccd3e8c7a8dc736a2dc`，与 #1 相同。
- CKPT 和 CFG_OPTS 只能在服务器上核对。
- `tools/query_diag/` 中，除下表外，其余文件的 md5 与 #1 完全相同：

| 文件 | 状态 | #1 md5 | 现 md5 | 归属 |
| --- | --- | --- | --- | --- |
| `diag_oracle_query.py` | CHANGED | `228804043303cef4472d203ccdba9b92` | `f867d88613806dd417b92e26d11d6a8d` | M1/M2 |
| `run_diag.sh` | CHANGED | `e19bf9b412419d53c7012dddca109a13` | `43a11f769b0b33176ff072f3fad28f5f` | M4，另含 #1 收尾时加的日志持久化（E2-1） |
| `diag_summarize.py` | CHANGED | `a86cf0a379154f9f45aadeeb14057ebf` | `deaa94b35ec03a7700d369631c671be2` | LOG：#1 收尾时加的写 `summary.txt`（E2-1） |
| `diag_b0_decoder.py` | NEW | – | `637b63a51b97f5b92f957aae548bf3ed` | M3 |
| `diag_exec2_check.py` | NEW | – | `271af47bf76a1aabbc16f33c15fdd40c` | M5 |
| `oracle_breakdown.py` | NEW | – | `a2991c9afcaed026845affcd4824fd55` | M5（迁入，E2-2） |
| `exec02_env.sh` | NEW | – | `a5a7eac2a914544ceaa19ef80f11683e` | M4 |

`run_diag.sh` 和 `diag_summarize.py` 的日志改动是 #1 第 3 步之后应用户要求加的。按 E2-1，它们在 S0' 允许范围内，条件是不改变任何计算结果（证据见 `EXEC02_DECISIONS.md`）。

## 6. 部署与运行

服务器代码按 CLAUDE.md 铁律 16 来自 zip。工作区 `$WS` 沿用 #1 的那份（A-1），其中是 #1 时的代码；本次的新代码要从本分支**最新**的 zip 取得。

注意：GitHub 分支 zip 的顶层目录名就是 `mmdetection3d-claude-dome-detr-query-diag`，与 `$WS` 同名。**不要在 `/212022085500129/dome-detr/` 下直接 unzip**，否则会解压进 `$WS` 本身。应解压到一个新建的空目录，`<解压目录>` 就是其中的 `mmdetection3d-claude-dome-detr-query-diag`。

```bash
# 0) 部署：新 zip 解压到新建的空目录（在 EP-Fusion 仓库和 $WS 之外），
#    只把 tools/query_diag/ 和 VERSION 拷进 $WS。
#    不动 work_dirs/、exec01_inputs、data 软链接和已编译的算子。
SRC=/212022085500129/dome-detr/_src_exec02
mkdir -p $SRC && unzip -q <新 zip 的路径> -d $SRC
NEW=$SRC/mmdetection3d-claude-dome-detr-query-diag     # 即 <解压目录>
head -1 $NEW/VERSION                                    # 应以 "2026-10-19 dome-detr exec02" 开头

WS=/212022085500129/dome-detr/mmdetection3d-claude-dome-detr-query-diag
cp -r $NEW/tools/query_diag/. $WS/tools/query_diag/
cp $NEW/VERSION $WS/VERSION
diff -rq -x __pycache__ $NEW/tools/query_diag $WS/tools/query_diag && echo synced
md5sum $WS/tools/query_diag/diag_oracle_query.py $WS/tools/query_diag/diag_exec2_check.py
#    应为 f867d886... / 271af47b...（第 5 节）

cd $WS
source tools/query_diag/exec02_env.sh  # 末行: [exec02_env] BASE_MAP=0.665463 BASE_NDS=0.706154 ...

# 1) 第 1 步（不占 GPU）：S0' -> B0 / S9
bash tools/query_diag/run_diag.sh b0
#    回传：work_dirs/diag/exec2/s0p_check.txt、b0_decoder.txt、
#          b0_decoder.json，以及 work_dirs/diag/logs/b0_*.log

# 2) 第 2 步（单卡，约 2.5 GPU 小时）：先确认空闲卡
nvidia-smi
CUDA_VISIBLE_DEVICES=<空闲卡号> bash tools/query_diag/run_diag.sh exec2
#    建议在 tmux / screen 中运行，防止终端断开时中断进程；
#    完整终端输出同时写入 work_dirs/diag/logs/exec2_*.log
```

第 2 步回传材料（对应本文档第 9 节）：

- `work_dirs/diag/exec2/oracle_e0_base/oracle_e0_base.json`、`s5p_gate.json`
- `work_dirs/diag/exec2/oracle_e{1,2,3}_*/oracle_*.json`
- `work_dirs/diag/exec2/exec2_summary.txt`（含 breakdown）和 `.json`
- `work_dirs/diag/logs/exec2_*.log`（各组的 `runtime` 字段在各自 json 内）

S0'、S9 或 S5' 任一不通过，脚本会自行停止（退出码 4），按第 7 节直接回传，不改口径。

## 7. 约束核对

- `projects/BEVFusion` 未改动；只读使用共享 conda 环境；不训练。
- 漏检定义、`--miss-thr` 2.0、jitter 1 格、seed 0 都固定在 `run_diag.sh exec2` 中，与本文档逐字一致。
- `exec2` 只跑 E0–E3，没有其他 oracle 变体。
- 单卡运行，C5 在每组启动前都会执行；S5' 不通过不会继续。
- 输出只写入 `work_dirs/diag/exec2/` 和带时间戳的 `work_dirs/diag/logs/`，不覆盖 #1 的结果。
