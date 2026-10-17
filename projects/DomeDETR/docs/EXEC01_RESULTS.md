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

## 第 1 步 部署门禁与 300 帧冒烟（待回传）

在工作区根目录执行，先 `source tools/query_diag/exec01_env.sh`。

```bash
for f in tools/query_diag/*.py; do python -c "import ast,sys; s=open(sys.argv[1],'rb').read(); assert all(b < 128 for b in s), 'non-ASCII'; ast.parse(s.decode(), feature_version=(3, 8)); print('OK', sys.argv[1])" "$f"; done
for f in tools/query_diag/*.sh; do bash -n "$f" && echo "bash -n OK $f"; done
CUDA_VISIBLE_DEVICES=<空闲卡> bash tools/query_diag/run_diag.sh quick
bash tools/query_diag/run_diag.sh merge
```

通过条件：`merge` 输出里 S1、S2、S3（a、b）、S4a、S4 全部 PASS。任一项不通过就停下回传，不调整统计口径。

需要回传：
- 门禁输出
- `work_dirs/diag/recall_quick/recall_report.txt`
- `work_dirs/diag/recall_quick/recall_merged.txt`
- 终端里 `[gpu-check]` 和 `[diag] runtime:` 这两行（耗时和峰值显存，用于估算全量时间）
