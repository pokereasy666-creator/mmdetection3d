# 执行文档 #1 v1.1 · C1–C6 实施记录与服务器运行手册

- 日期：2026-10-17
- 分支：`claude/dome-detr-query-diag`
- 参考实现原样快照：提交 `0ca9cf6`（`tools/query_diag/` = `query_diag.tar.gz` 原文）。本节所有 diff 都相对这个提交。
- 执行环境：云端 CPU 容器。服务器上的 S0–S3 **都还没运行**，第 5 节是运行手册。

## 1. diff 摘要

```
tools/query_diag/diag_common.py       |  C1 快照、C6 runtime 工具函数
tools/query_diag/diag_query_recall.py |  C1 T4 改读快照、S4a 统计、C6、fp16 跟随 test_cfg
tools/query_diag/diag_oracle_query.py |  C3 提示、C6
tools/query_diag/diag_summarize.py    |  S5/S6/S7 PASS/FAIL 行（清单外，见下）
tools/query_diag/run_diag.sh          |  C2、C3、C5，新增 s0 / merge / s5-adjudicate 入口
新增 diag_gate_s5.py                  |  C2
新增 diag_merge_bins.py               |  C4
新增 diag_s0_check.py + s0_manifest_82c2b4e.json | S0a–S0d（v1.1 第 5 节）
```

| 编号 | 改了什么 |
| --- | --- |
| C1 | `install_capture` 在替换后的 `forward` 里、`predict_by_feat` 之前，执行 `STATE['center_grid'] = center[..., -P:].detach().clone()`。T4 改读这份快照。recall 循环统计快照里落在 [−2, H+2] 的比例，写入 `recall_report.txt` 首部和 `recall_summary.json` 的 `center_grid_in_range_frac`，供 S4a 使用。 |
| C2 | `oracle` 开跑前必须有 `BASE_MAP` / `BASE_NDS`（没设置就直接退出，不先跑 base）。流程是：base → `diag_gate_s5.py gate`（每项 \|Δ\| ≤ 0.001，另加 1e-9 浮点余量）→ 不通过就以退出码 4 结束，不跑后四组。新增 `s5-adjudicate`：单卡跑无 hook 的 `tools/test.py`，再用 `diag_gate_s5.py adjudicate`（容差 0.0005）给出 SOURCE / HOOK 判定。两种判定都以退出码 4 结束，脚本不会自动继续，也不会替换参考值。 |
| C3 | `CONFIG` 改为必填（去掉了默认的官方配置）。`CFG_OPTS` 透传给 `--cfg-options`。非空时：终端打印 NOTE；json 的 `runtime.cfg_options` 与 `runtime.cfg_opts_env` 都有记录；`summary` 对每组单独打印 NOTE。 |
| C4 | `diag_merge_bins.py` 和 `merge` 入口，会处理 `recall_quick` 和 `recall` 两个目录中已存在的那些。输出 v1.1 第 7 节的全部量：R_near/R_far(200/1000)、miss、比值、ΔR_far、agn_gap_far、final_gap_far、frac_over_200、两段 GT 数。只对 S1、S2、S3（a：无 `[WARNING]`；b：0–20 m 段 ≥ 0.5）、S4a、S4 给 PASS/FAIL，不输出任何 R 规则判定。 |
| C5 | 必选，没有跳过开关。每次启动 GPU 进程前（包括 oracle 每一组）检查 `CUDA_VISIBLE_DEVICES` 里的每张卡：只要有计算进程，或 `memory.used` > 1024 MiB，就以退出码 3 拒绝启动；找不到 `nvidia-smi`、没设 `CUDA_VISIBLE_DEVICES`、卡数不对（quick / recall / s5-adjudicate 要求恰好 1 张，oracle 要求恰好 `GPUS` 张且只能是 1 或 2），都拒绝启动。同时导出 `CUDA_DEVICE_ORDER=PCI_BUS_ID`，保证 CUDA 编号与 `nvidia-smi` 编号一致。 |
| C6 | `recall_summary.json` 和每个 `oracle_*.json` 都带 `runtime`，内容包括：<br>• 总耗时，以及峰值显存 allocated / reserved（MiB）、GPU 型号<br>• python、torch、mmcv、mmengine、mmdet、mmdet3d、numpy、torch_cuda 的版本<br>• 仓库 `VERSION` 第一行<br>• CONFIG 和 CKPT 的路径与 md5<br>• `fp16`：`val_cfg` / `test_cfg` 各自的设置，recall 另记 `recall_loop_fp16`<br>• `cfg_options`、`CUDA_VISIBLE_DEVICES`、`world_size`<br>• `diag_files_md5`（清单外，见下） |

**行为上的唯一变化**：recall 循环现在把 `test_step` 包在 `mmengine.runner.amp.autocast(enabled=test_cfg.fp16)` 里，写法与官方 `TestLoop` 相同。当前配置 `test_cfg = dict()`，即 fp16=False，所以数值不变。这样做是为了让 D1 与 D2 和 `tools/test.py` 走同一套精度设置。

**超出 C1–C6 清单的改动（请确认）**：
- `diag_summarize.py` 增加 S5/S6/S7 的 PASS/FAIL 行。只有 S 类，R 类仍留给审阅。
- `diag_s0_check.py` 和 `s0_manifest_82c2b4e.json`：v1.1 第 5 节要求在服务器上做 S0，必须有脚本才能执行，设计见第 4 节。
- `runtime.diag_files_md5`：服务器上的诊断代码是手工拷过去的，而仓库 `VERSION` 早于这批代码，所以单看 `VERSION` 无法确认实际运行的是哪一版。这里记录每个诊断文件的 md5。
- `projects/DomeDETR/tests/`：开发侧 CPU 测试，**不部署**到服务器。

## 2. 同步门禁输出（本地，Python 3.13，按 3.8 语法解析）

```
ast(3.8) OK  tools/query_diag/diag_common.py
ast(3.8) OK  tools/query_diag/diag_gate_s5.py
ast(3.8) OK  tools/query_diag/diag_merge_bins.py
ast(3.8) OK  tools/query_diag/diag_oracle_query.py
ast(3.8) OK  tools/query_diag/diag_query_recall.py
ast(3.8) OK  tools/query_diag/diag_s0_check.py
ast(3.8) OK  tools/query_diag/diag_summarize.py
bash -n OK  tools/query_diag/run_diag.sh
non-ASCII lines: 0 in every file under tools/query_diag/
```

部署后需要在服务器上（Python 3.8）重跑一次，命令见第 5 节。

## 3. 本地 CPU 验证（mock）

`projects/DomeDETR/tests/test_query_diag_cpu.py`，13 项全部通过。

- mock head 的以下部分**逐行复制**自本仓库，包括原地改写 `center` 的那几行：
  - query 初始化（`transfusion_head.py:227-262`）
  - `query_heatmap_score`（:298-304）
  - `predict_by_feat` 的分数计算与 `center` 切片（:358-386）
  - `TransFusionBBoxCoder.decode`（`utils.py:72-98`）
- 只有骨干网络和解码器换成了合成替身。
- 测试驱动的是真实的 `diag_query_recall.main()` 和 `diag_oracle_query.main()`，只把 `Runner` 和 `Config.fromfile` 替换成假对象。

| 测试 | 关键结果 |
| --- | --- |
| 参考实现，复现 F1 | T4 0–20 m：`init_agn` = 0.943，`final_agn` = **0.000**，S4 失败 |
| 新实现，C1 | S1 匹配率 = 1.0，`transposed` = False，S4a = 1.0，`final_agn` 与 `init_agn` 相同 |
| merge | 对 S 项全部判 PASS；所有规则输入量与直接用 npz 重算的结果一致（误差 < 1e-12）；不输出 R 判定。对参考实现的输出跑 merge 时，S2、S4a 判 MISSING，S4 判 FAIL |
| oracle（loc / loc_cls / far） | `init_mismatch_rate` = 0，`capped_cells` = 0，`restored_queries` = `injected_cells`；被注入的 query 的 `query_heatmap_score` 逐个等于原始 pre-NMS sigmoid，没有泄漏置信度 |
| S5 gate | 0.6660 对 0.6650 判 PASS（浮点余量起作用），差 0.0011 判 FAIL |
| S5 裁决 | SOURCE 和 HOOK 两种判定、json 来源和文本日志回退都覆盖到 |
| run_diag.sh（替换了 `nvidia-smi` 和 `python`） | 没设 CONFIG 时拒绝；没设 `CUDA_VISIBLE_DEVICES` 时拒绝；有进程时退出码 3；显存 2048 MiB 时退出码 3，1024 MiB 时放行；卡数不对时拒绝；S5 不通过时只跑了 base，退出码 4；S5 通过时按 base → all → far → near → all_cls 顺序全部执行；`GPUS=3` 被拒绝；`CFG_OPTS` 透传到参数末尾 |
| S0a | 本仓库对照清单全部 OK；改动核心文件判 FAIL；未跟踪的 .py 判 FAIL；.so 被忽略；改动 .md 只判 WARN；import 解析到仓库外判 FAIL |
| S0b | 两份真实配置的覆盖项共 14 条（见第 4 节），`model` 类型集合相同 |
| S0d | 评测设置比较正确；`pretty_text` 来回转换后无差异；只改训练项时不报差异；改测试项时报 3 处差异 |

**这套测试不能证明的**：
- 真实模型上的数值。mock 只验证代码路径和统计逻辑。
- DDP 路径。v1.1 建议 `GPUS=1`。
- S0b 的注册来源检查，以及 S0c、S0d 的完整路径。这些需要 mmcv、mmdet3d 和 CUDA 算子，本地只测了不依赖它们的部分（配置 diff、评测设置比较、`pretty_text` 来回转换）。

## 4. S0 的实现与判据对应

| 自检 | 实现 | 与 v1.1 文字的差别 |
| --- | --- | --- |
| S0a | 用 `82c2b4e` 生成的 md5 清单（`s0_manifest_82c2b4e.json`）覆盖 `projects/BEVFusion` 下全部 33 个已跟踪文件。源码和配置有差异 → FAIL；`.md` 有差异 → **WARN**；该目录下多出未跟踪的源码文件 → FAIL；编译产物（.so、build、egg-info、`__pycache__`）忽略。另外核对 `mmdet3d` 和 `projects.BEVFusion` 实际是从本仓库 import 的，否则 FAIL。审计引用过的 mmdet3d 文件、`tools/test.py`、`default_runtime.py` 作为扩展集，只给 WARN。 | 不调用 git：CLAUDE.md 铁律 16 规定服务器代码来自 zip。md5 清单覆盖整个目录，比 v1.1 列出的 7 个文件更严。**`.md` 只判 WARN，请确认是否接受。** |
| S0b | 用 mmengine 加载两份配置后递归 diff，逐条列出覆盖项。`model` 中出现官方配置没有的 `type` 时 FAIL；`custom_imports` 不同时 FAIL；在注册表里查每个 model type 实际来自哪个模块，模块不在官方前缀内或文件在仓库外时 FAIL。 | — |
| S0c | 在 CPU 上构建模型，用 mmengine 的 `load_checkpoint(strict=False)` 加载权重（与 `Runner.load_checkpoint` 同一调用链），并挂一个 logger 收集它的输出，在其中查找 `missing keys`、`unexpected key`、`size mismatch`。另外直接比对 key 和形状，只作参考。 | 需要专门收集日志的原因：`Runner.load_checkpoint` 调用 `print_log(logger=None)`，不匹配信息只打到 stdout，**不会进日志文件**。 |
| S0d | 检查 val 与 test 的 dataloader.dataset 和 evaluator 是否完全相同，记录并比较 fp16。用权重里 mmengine 保存的训练配置（`meta['cfg']`）比较三处：val 数据集 / evaluator / `val_cfg.fp16`，以及去掉 `init_cfg`、`train_cfg` 后的 `model`。给了 `TRAIN_WORK_DIR` 时，再比较当时转存的配置，并从 `vis_data/scalars.json` 列出每个 epoch 的 val mAP/NDS（6 位小数），标出与权重 `meta.epoch` 对应的那一行。 | BASE_MAP / BASE_NDS 仍由你填写，脚本只把日志里的值列出来。 |

S0b 在 `82c2b4e` 上的预期输出共 14 条，服务器结果应与此完全一致：

```
model.img_backbone.init_cfg.checkpoint: <github URL> -> '/212022085500129/mmdetection3d-claude-jolly-wozniak-c4YMQ/checkpoints/swint-nuimages-pretrained.pth'
optim_wrapper.accumulative_counts: <absent> -> 4
optim_wrapper.clip_grad.max_norm: 35 -> 10
optim_wrapper.loss_scale: <absent> -> 512.0
optim_wrapper.type: 'OptimWrapper' -> 'AmpOptimWrapper'
param_scheduler[1].T_max: 6 -> 20
param_scheduler[1].end: 6 -> 20
param_scheduler[2].end: 2.4 -> 8
param_scheduler[3].begin: 2.4 -> 8
param_scheduler[3].end: 6 -> 20
train_cfg.max_epochs: 6 -> 20
train_dataloader.batch_size: 4 -> 2
train_dataloader.dataset.dataset.pipeline[10].max_epoch: 6 -> 20
train_pipeline[10].max_epoch: 6 -> 20
```

`model` 下唯一的覆盖项是 Swin 初始化路径。这个路径在加载完整权重后会被覆盖，不影响推理。test / val 相关的项没有任何覆盖。

## 5. 服务器运行手册

在服务器仓库根目录、训练用的同一个 conda 环境中执行。分支 zip 先解压到**仓库外**，例如 `~/qd_zip`。

```bash
cd /212022085500129/mmdetection3d-claude-jolly-wozniak-c4YMQ
export CONFIG=projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_4xa30-amp-accum_nus-3d.py
export CKPT=<epoch_19.pth 的绝对路径>

# S0：只用 CPU，直接从 zip 目录运行，此时仓库里还没有 tools/query_diag/
TRAIN_WORK_DIR=work_dirs/baseline1 bash ~/qd_zip/tools/query_diag/run_diag.sh s0
#   -> work_dirs/diag/s0/s0_report.txt。S0a–S0d 全部 PASS 才继续；否则停下回传。

# 第 1 步：部署、门禁、冒烟
cp -r ~/qd_zip/tools/query_diag tools/
for f in tools/query_diag/*.py; do python -c "import ast,sys; s=open(sys.argv[1],'rb').read(); assert all(b < 128 for b in s), 'non-ASCII'; ast.parse(s.decode(), feature_version=(3, 8)); print('OK', sys.argv[1])" "$f"; done
bash -n tools/query_diag/run_diag.sh && echo 'bash -n OK'
CUDA_VISIBLE_DEVICES=<空闲卡> bash tools/query_diag/run_diag.sh quick
bash tools/query_diag/run_diag.sh merge        # S1–S4a 全部 PASS 才继续

# 第 2 步：全量 D1 + D3
CUDA_VISIBLE_DEVICES=<空闲卡> bash tools/query_diag/run_diag.sh recall
bash tools/query_diag/run_diag.sh merge

# 第 3 步：D2（R1 阈值需在此前确定）
BASE_MAP=<4位> BASE_NDS=<4位> GPUS=1 CUDA_VISIBLE_DEVICES=<空闲卡> \
  bash tools/query_diag/run_diag.sh oracle
bash tools/query_diag/run_diag.sh summary
# 只有 S5 不通过时才执行：
CUDA_VISIBLE_DEVICES=<空闲卡> bash tools/query_diag/run_diag.sh s5-adjudicate
```

退出码：0 = 正常；2 = 参数或环境不满足；3 = GPU 被占用；4 = S5 不通过，或 S5 裁决已结束（需要人工确认）。

## 6. 回传对照

| v1.1 第 9 节 | 文件 |
| --- | --- |
| S0a–S0d | `work_dirs/diag/s0/s0_report.txt`（详情在同目录 json） |
| 第 1 步 | `work_dirs/diag/recall_quick/recall_report.txt`、`recall_merged.txt`，终端里的 runtime 行 |
| 第 2 步 | `work_dirs/diag/recall/recall_report.txt`、`recall_summary.json`、`recall_merged.txt` |
| 第 3 步 | `work_dirs/diag/oracle_*/oracle_*.json`、`oracle_base/s5_gate.json`、`summary` 的完整输出；触发裁决时附 `s5_nohook/s5_adjudication.json` |
| runtime | 上述 json 中的 `runtime` 字段 |
