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

## 第 2 步 全量 D1 + D3（待回传）

在工作区根目录执行。耗时较长，建议放在 `tmux` 或 `screen` 里跑，避免 SSH 断开时进程被杀。

```bash
source tools/query_diag/exec01_env.sh
for f in tools/query_diag/*.py; do python -c "import ast,sys; s=open(sys.argv[1],'rb').read(); assert all(b < 128 for b in s), 'non-ASCII'; ast.parse(s.decode(), feature_version=(3, 8)); print('OK', sys.argv[1])" "$f"; done
for f in tools/query_diag/*.sh; do bash -n "$f" && echo "bash -n OK $f"; done
CUDA_VISIBLE_DEVICES=<空闲卡> bash tools/query_diag/run_diag.sh recall
bash tools/query_diag/run_diag.sh merge
```

参数固定为 v1.1 第 7 节的取值，不得调整。

需要回传：
- 门禁输出（补交第 1 步缺的那份）
- `work_dirs/diag/recall/recall_report.txt`
- `work_dirs/diag/recall/recall_summary.json`
- `work_dirs/diag/recall/recall_merged.txt`
- 终端里的 `[gpu-check]` 行和 `[diag] runtime:` 行

`merge` 会把 `recall_quick` 和 `recall` 两个目录都处理一遍。`recall_quick` 的结果与上次相同，不必再回传。
