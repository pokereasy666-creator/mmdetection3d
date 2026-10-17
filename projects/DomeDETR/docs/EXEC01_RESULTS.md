# 执行文档 #1 · 结果与审阅记录

每一步回传后在这里追加一节。原始报告存放在 `projects/DomeDETR/results/exec01/`。判读规则见 v1.1 第 10 节和 `EXEC01_DECISIONS.md`。

## S0 服务器一致性检查（2026-10-17 回传）：全部 PASS

- 原始报告：`results/exec01/s0_report.txt`
- 工作区：`/212022085500129/dome-detr/mmdetection3d-claude-dome-detr-query-diag`（A-1）

| 自检 | 结果 | 关键证据 |
| --- | --- | --- |
| S0a | PASS | 33/33 个已跟踪文件的 md5 与 `82c2b4e` 一致；没有未跟踪的源码文件；扩展集 9/9 一致；`mmdet3d` 和 `projects.BEVFusion.bevfusion` 都从工作区导入。三份配置与审计版完全相同，D-3 没有触发 |
| S0b | PASS | 覆盖项共 14 条，与实施文档第 4 节的预测逐条相同；`model` 中的类型 28/28，没有新增类型 |
| S0c | PASS | 张量 582/582；加载日志里没有 missing keys、unexpected keys、size mismatch；直接比对的差异数为 0（稀疏卷积权重布局一致）；有相机分支；meta 中 epoch=5、iter=77240（= 15448 × 5）、seed=577127641 |
| S0d | PASS | val 与 test 使用同一个标注文件和同样的 evaluator；fp16 两边都是 False；与 2 份训练配置（权重内嵌的那份和训练目录里转存的那份）比较，评测相关部分没有任何差异；BASE 核对：第 5 轮 val 全精度 mAP 0.664817、NDS 0.706014，取 4 位后正好是 0.6648 / 0.7060，与 D-2 一致 |

**审阅备注**：
1. **CUDA 算子是在工作区内编译的。** S0a 的 import 检查要先加载 `projects.BEVFusion.bevfusion.ops`，需要算子的 .so 文件；这一步成功，而且解析路径在工作区内，说明 `build-ops` 生效，没有借用 EP-Fusion 仓库里的编译产物。
2. **训练时用的配置文件名与本次不同。** 权重 meta 中的 `experiment_name` 是 `bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_20260718_020825`，说明那次训练是用官方融合配置的文件名、再加覆盖参数启动的，不是 4xa30 配置文件。S0d 已经拿这次实际使用的 CONFIG 与权重内嵌配置、训练目录转存配置分别比较过，评测相关部分完全一致，所以不影响结论。这里只作为来源信息记录。
3. **`CFG_OPTS` 生效**（按 C3 说明）：Swin 初始化路径被钉到工作区内的拷贝。完整权重加载后会覆盖这部分权重，不影响推理。

**结论**：S0 通过，进入第 1 步。

## 第 1 步 300 帧冒烟（2026-10-17 回传）：S1–S4a 全部 PASS

- 原始输出：`results/exec01/step1_recall_quick_report.txt`、`step1_recall_quick_merged.txt`、`step1_terminal.txt`
- GPU 1：检查时空闲（没有计算进程，`memory.used` = 0 MiB）
- 耗时：总计 117.1 s，其中循环 92.3 s（300 帧，约 0.31 s/帧）
- 峰值显存：allocated 2231.8 MiB，reserved 3276.0 MiB

| 自检 | 值 | 判据 | 结果 |
| --- | --- | --- | --- |
| S1 初始化复现与 `head.query_labels` 匹配率 | 1.00000 | ≥ 0.99 | PASS |
| S2 `bev_pos transposed` | False | == False | PASS |
| S3a 报告中 `[WARNING]` 行数 | 0 | == 0 | PASS |
| S3b 0–20 m 段 class-aware 初始召回率 @2 m，K=200 | 0.98781 | ≥ 0.5 | PASS |
| S4a `center_grid` 落在 [−2, H+2] 的比例 | 0.99833 | ≥ 0.99 | PASS |
| S4 0–20 m 段 final_agn − init_agn | +0.00037 | ≥ −0.05 | PASS |

**审阅备注**：
1. **C1 在真实模型上生效。** T4 中 final 与 init 几乎相等（全体样本均为 0.980），而参考实现在 mock 上会塌到约 0。
2. **S4a 的 0.17% 越界在预期之内。** 120000 个中心坐标里约 200 个落在 [−2, 182] 之外，来自网格边缘附近、回归偏移较大的 query，远低于 1% 的门槛。
3. **坐标约定正确。** 0–20 m 段 class-agnostic 初始召回率 @4 m 为 0.997，若坐标轴颠倒，这个值会远低于 0.5。
4. **门禁输出没有回传。** v1.1 第 9 节要求提供。所有文件在同步前已经在本地通过门禁（见实施文档第 2 节），冒烟实际执行过的脚本也在服务器的 Python 3.8 上正常运行；但 `diag_oracle_query.py`、`diag_gate_s5.py`、`diag_summarize.py` 还没在服务器上跑过。要求在第 3 步之前补交服务器上的门禁输出。
5. **冒烟数字只用于自检。** 300 帧样本太小（例如 motorcycle 只有 10 个 GT，trailer 为 0），不用于任何 R 规则，也不据此调整任何后续步骤。

**结论**：第 1 步通过，进入第 2 步。全量 6019 帧的循环预计约 31 分钟，加上模型构建，共约 33 分钟。

## 第 2 步 全量 D1 + D3（2026-10-17 回传）：S1–S4a 全部 PASS

- 原始输出：`results/exec01/step2_recall_report.txt`、`step2_recall_summary.json`、`step2_recall_merged.txt`、`step2_terminal.txt`（门禁和运行行的节选）
- 门禁：服务器 Python 3.8 上 7 个 .py 和 3 个 .sh 全部通过，补齐了第 1 步缺的那份
- GPU 1：检查时空闲
- 耗时：总计 1815 s，循环 1792.6 s（6019 帧，0.298 s/帧）
- 峰值显存：allocated 2233.8 MiB，reserved 3276.0 MiB
- 运行环境（runtime 字段）：torch 2.0.1+cu118、mmcv 2.1.0、mmengine 0.10.5、mmdet 3.2.0、mmdet3d 1.4.0；`config_md5` 8f49ac79…，`checkpoint_md5` c16536e3…，均与 S0 一致；fp16 两边都是 False；`CFG_OPTS` 为 Swin 路径（C3）

### 自检

| 自检 | 值 | 判据 | 结果 |
| --- | --- | --- | --- |
| S1 | 1.00000 | ≥ 0.99 | PASS |
| S2 | False | == False | PASS |
| S3a | 0 | == 0 | PASS |
| S3b（0–20 m，class-aware，@2 m，K=200） | 0.97621 | ≥ 0.5 | PASS |
| S4a | 0.99927（共 2,407,600 个值） | ≥ 0.99 | PASS |
| S4 | +0.00048 | ≥ −0.05 | PASS |

### 第 7 节规则输入量

除注明外均为 class-aware、@2 m。共 118,538 个 GT：<30 m 段 86,121 个，≥30 m 段 32,417 个。

| 量 | 值 | 二项标准误* |
| --- | --- | --- |
| R_near(200) / R_near(1000) | 0.97319 / 0.98742 | — |
| R_far(200) / R_far(1000) | 0.94944 / 0.97273 | — |
| miss_near / miss_far | 0.02681 / 0.05056 | 0.00055 / 0.00122 |
| miss_far / miss_near | **1.886** | 0.060 |
| ΔR_far | **0.0233** | 0.0008 |
| agn_gap_far | **0.0247** | 0.0009 |
| final_gap_far | **0.0002** | — |
| frac_over_200 | **0.0000**（每帧最多 102 个，p99 = 76.8） | — |

\* 按 GT 相互独立计算。同一帧内的 GT 相关，真实误差会更大，这里只用来看数量级。

### 与阈值的相对位置（不是判读）

**最终判读按第 10 节的顺序进行，第一条 R1 需要第 3 步的 ΔmAP_all，所以下表不构成任何结论，只记录 D1 / D3 的数字落在哪个区间。**

| 规则 | 只依赖 D1 / D3 的部分 | 落点 |
| --- | --- | --- |
| R2 | miss_far / miss_near ≥ 2 | 1.886 < 2，这半条不满足。R2 要求两个条件同时成立，因此无论 ΔmAP_far 是多少，R2 都不会满足 |
| R3 | ΔR_far 位于 ≥ 0.05 / 0.02–0.05 / ≤ 0.02 哪一段 | 0.0233，落在 0.02–0.05 的"审阅讨论"区间，比下界 0.02 高约 4 个标准误 |
| R4 | agn_gap_far ≥ 0.05（另一个分支需要 D2） | 0.0247 < 0.05，这个分支不触发；ΔmAP_all_cls − ΔmAP_all 那个分支待第 3 步 |
| R5 | final_gap_far ≥ 0.05 | 0.0002，不满足：解码器没有补回初始化漏掉的目标 |
| R6 | frac_over_200 < 0.5% | 0，满足 |

### 审阅备注（事实层面）

1. **初始化召回整体已经很高。** K=200、2 m 内有同类 query 的 GT 占 96.7%（不区分类别时为 98.2%）；≥30 m 段为 94.9%（不区分类别 97.4%）。
2. **漏检里有近一半是类别锁定造成的。** 全体漏检 3.33% 中约 45%、远距离漏检 5.06% 中约 49% 属于"附近有 query，但初始化时类别判错"（class-agnostic 和 class-aware 之差）。agn_gap_far 本身（0.0247）仍低于 R4 的 0.05。
3. **加预算能找回一部分。** K 从 200 增加到 1000，远距离 class-aware 召回 +2.3 pp，class-agnostic +1.9 pp（0.974 → 0.993）。
4. **漏检集中在少数类别，且与距离关系不一。** class-aware @2 m 全距离：construction_vehicle 0.716、trailer 0.857、bicycle 0.909、truck 0.935、bus 0.943、barrier 0.943。barrier 在 0–20 m 段也只有 0.942，说明不全是距离问题。
5. **解码器只做局部细化，不改变召回。** T4 中 final 与 init 在各段之间的差都在 0.002 以内（R5 的落点也是这个意思）。
6. **D3**：每帧有效 GT 最多 102 个，没有任何一帧超过 200 个。

**结论**：第 2 步通过，可以进入第 3 步。按预登记的流程，D1 的数字不改变第 3 步的执行方式。

## 第 3 步 D2 oracle 五组（待回传）

在 `tmux` 里执行。先跑 base，S5 通过后会自动接着跑另外 4 组；每组约 40 分钟（推理约 30 分钟 + nuScenes 评测约 9 分钟），共约 3.3–3.5 小时。

```bash
source tools/query_diag/exec01_env.sh
GPUS=1 CUDA_VISIBLE_DEVICES=<空闲卡> bash tools/query_diag/run_diag.sh oracle
bash tools/query_diag/run_diag.sh summary
```

- 退出码 4 且提示 `S5 FAILED`：后四组不会运行。按预登记流程执行 `CUDA_VISIBLE_DEVICES=<空闲卡> bash tools/query_diag/run_diag.sh s5-adjudicate`，回传结果后**停下等确认**。
- 中途若因 GPU 被占用（退出码 3）而停止：已完成的组的 json 会保留，但重新执行 `oracle` 会从 base 开始重跑。

需要回传：
- `work_dirs/diag/oracle_{base,all,far,near,all_cls}/oracle_*.json`（5 个）
- `work_dirs/diag/oracle_base/s5_gate.json`
- `summary` 的完整输出
- 终端里的 `[gpu-check]`、`[S5]`、`[oracle]` 开头的行
