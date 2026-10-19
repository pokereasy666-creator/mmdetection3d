# 执行文档 #2a 实施说明（E3'）

- 执行文档：`exec_doc_02a_E3prime.md`（v1.1，本目录）
- 分支：`claude/dome-detr-query-diag`
- 状态：工具与开发侧测试完成；服务器运行待执行

## 1. 实现方式与一处需要确认的处理

#2a 第 2 节要求"不改代码，只用现有参数"，S0'' 要求 `tools/query_diag/` 全部文件的 md5 与 #2 第 2 步相同。但现有工具里有两处缺口：`run_diag.sh` 没有 E3' 的入口，#2a 第 4 节要求的汇总量（L、保留率'、r_0.5、S8'）也没有现成的计算。

执行侧的处理：

- **`tools/query_diag/` 一个字节都不改。** 两组 oracle 由未改动的 `diag_oracle_query.py` 运行，参数与 #2a 第 2 节逐字相同。
- **新增独立目录 `tools/query_diag_exec2a/`**，里面只有运行器和汇总，不参与推理：

| 文件 | md5 | 作用 |
| --- | --- | --- |
| `run_exec2a.sh` | `2abe3ae9…` | 入口 `s0pp`、`run`、`summary`。A-1 受保护目录检查、C5 GPU 检查、`CFG_OPTS` 处理都从 `run_diag.sh` 逐字复制（有测试核对），固定单卡，不用 torchrun。日志写到 `work_dirs/diag/logs/exec2a-<入口>_<时间>.log` |
| `exec2a_check.py` | `3dd7e52a…` | 子命令 `s0pp`（S0''）、`check-run`（单组的 S6 / S7' / S8' 和参数核对）、`summary`（汇总）。breakdown 直接调用未改动的 `tools/query_diag/oracle_breakdown.py` |

**需要确认**：我把"不改代码"理解为"不改 oracle 代码和 `tools/query_diag/`"，运行器和汇总放在新目录中。如果你的本意是连新目录也不加，那就只能手工跑两条 python 命令，C5 检查和日志也得靠手工完成。

## 2. `run` 的流程

1. **前置条件**：`work_dirs/diag/exec2/oracle_e0_base/oracle_e0_base.json` 和 `work_dirs/diag/exec2/b0_decoder.json` 都存在，否则拒绝启动。
2. **S0''**：`tools/query_diag/` 全部 `.py/.sh/.json`（15 个）、CONFIG md5、CKPT md5、`CFG_OPTS` 都与 #2 第 2 步四组 json 的 runtime 记录相同；四组记录之间也必须一致。新目录本身要过门禁（非 ASCII、Python 3.8 语法、`bash -n`）。不通过则退出（退出码 4），不占 GPU。
3. **E3'-exact**：先做 C5 检查，再运行 `--oracle loc --subset missed --miss-thr 0.5 --range 0 1000 --score orig`。
4. **E3'-exact 自检**：
   - ARGS：参数与 #2a 第 2 节完全一致
   - S6
   - S7'：selected ≥ 0.99 × injected，且 restored = selected
   - S8'：`missed_gt` 与 B0 子集规模的相对差 ≤ 1%
   
   不通过就停（退出码 4）：E3'-jitter 不运行，但会写出汇总以便回传。
5. **E3'-jitter**：再做一次 C5 检查，然后运行同一组参数加 `--jitter-cells 1 --seed 0`。
6. **汇总**：写入 `work_dirs/diag/exec2a/exec2a_summary.txt` / `.json`。任一自检不通过时退出码为 4。

## 3. 汇总内容（不计算 Q4 / Q3' 判定）

**基准与检查**

- E0 核对：参照文件必须是 #2 的 E0，mAP / NDS 取 6 位小数后为 0.666180 / 0.706324。
- S8' 参照量 = N_gt × (1 − B0 中 0.5 m 的 class-aware 初始召回) = 118538 × (1 − 0.82792) = **20398**。另外核对两组之间的相对差 ≤ 1%。如果 `--miss-thr 0.5` 没有生效，`missed_gt` 会接近 2 m 口径的 3948，S8' 会直接失败。

**主要量**

- 两组各自相对 E0 的 ΔmAP / ΔNDS、mATE、mAOE。
- 各阈值的 ΔmAP 贡献：类平均 ΔAP ÷ 4。
- exact_gain、jitter_gain。
- L = exact_gain − 0.300（0.300 为登记值）。
- 保留率' = jitter_gain ÷ exact_gain，只在 exact_gain ≥ 0.5 pp 时计算，负值照原样输出。
- r_0.5 = 两组 0.5 m 贡献之比。

**明细与附录**

- 分类别、分阈值的 ΔAP。做 3×3 local-max 的 8 类与不做 NMS 的对照类（pedestrian、traffic_cone）分开列出。
- `oracle_breakdown.py` 全文。
- 各组的 runtime 字段和日志路径。

## 4. 开发侧测试（`tests/test_exec2a_cpu.py`，6/6 PASS）

1. **汇总的算术**：用真实的 E0、B0 和人为构造的 E3' json 核对 L、保留率'、r_0.5、各阈值贡献、子集规模 20398、对照类单列；汇总中不含判定用语。
2. **汇总的负例**：
   - `missed_gt` 偏 2% 时 S8' 判 FAIL
   - `miss_thr` 写成 2.0 时 ARGS 判 FAIL
   - 参照文件换成 E1 时 E0 核对判 FAIL
   - exact_gain < 0.5 pp 时不计算保留率'
3. **在 mock 上用未改动的 oracle 代码跑两组**，`check-run` 通过，`missed_gt` 等于 D1 的 `rank_ca@0.5 ≥ P` 计数。局限：mock 的初始峰值都在真值格上，0.5 m 和 2 m 子集恰好重合，所以 0.5 m 阈值本身由服务器上的 S8' 验证。
4. **S0''**：以 #2 第 2 步的真实 runtime md5 为参照，本仓库的 `tools/query_diag/` 判 PASS。以下情况判 FAIL：CFG_OPTS 不同、文件被改动、多出文件、四组 runtime 记录不一致。
5. **运行器**（桩 python / nvidia-smi）：
   - 正常情况下按 S0'' → exact → check-run → jitter → summary 的顺序执行，参数逐字核对，C5 执行 2 次，输出目录和日志名正确
   - S0'' 不通过：不跑任何 oracle
   - exact 自检不通过：不跑 jitter
   - 2 张卡或卡被占用：拒绝启动
   - 在受保护目录内：拒绝启动
6. **复制核对**：运行器中的 guard 函数与 `run_diag.sh` 逐字相同。

## 5. 部署与运行

工作区 `$WS` 不变。新 zip 要解压到新建的空目录（顶层目录与 `$WS` 同名，不要直接解压到 `/212022085500129/dome-detr/` 下），**只拷新目录和 VERSION，不拷 `tools/query_diag/`**（它必须保持 #2 第 2 步的原样，S0'' 会核对）。

```bash
SRC=/212022085500129/dome-detr/_src_exec02a
mkdir -p $SRC && unzip -q <新 zip 的路径> -d $SRC
NEW=$SRC/mmdetection3d-claude-dome-detr-query-diag
WS=/212022085500129/dome-detr/mmdetection3d-claude-dome-detr-query-diag
cp -r $NEW/tools/query_diag_exec2a $WS/tools/
cp $NEW/VERSION $WS/VERSION
md5sum $WS/tools/query_diag_exec2a/*        # 应与第 1 节的表一致

cd $WS
source tools/query_diag/exec02_env.sh
bash tools/query_diag_exec2a/run_exec2a.sh s0pp      # 可选：先单独看 S0''
nvidia-smi
CUDA_VISIBLE_DEVICES=<空闲卡号> bash tools/query_diag_exec2a/run_exec2a.sh run
```

约 1.3 GPU 小时，建议在 tmux 中运行。

回传材料：

- `work_dirs/diag/exec2a/` 下的 `s0pp_check.txt`、`exec2a_summary.txt`、`exec2a_summary.json`
- `oracle_e3p_exact/oracle_e3p_exact.json`、`oracle_e3p_jitter/oracle_e3p_jitter.json`
- `work_dirs/diag/logs/exec2a-run_*.log`

## 6. 约束核对

- `projects/BEVFusion` 和 `tools/query_diag/` 均未改动。
- `--miss-thr 0.5`、jitter 1 格、seed 0 都固定写在 `run_exec2a.sh` 中，没有其他变体，也不重跑 base。
- 单卡运行，每组启动前做 C5 检查。
- 输出只写入 `work_dirs/diag/exec2a/` 和带时间戳的日志，#1、#2 的结果不受影响。
