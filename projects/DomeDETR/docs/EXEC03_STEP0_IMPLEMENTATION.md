# 执行文档 #3 · 第 0 步实现（v1.1）

- 依据：`exec_doc_03_query_localization.md` v1.1（V1–V5），以及已确认的 `EXEC03_PLAN.md`（P-1 至 P-12、R-1 至 R-7）
- 范围：
  - `projects/QueryLoc` 的模型代码和配置；
  - 第 4.2 节的预演、第 4.3 节的 S-T0 / S-T2 / S-M0 / S-M1 工具；
  - 第 0b 步的评测入口。
- 未改动：`projects/BEVFusion/`、`tools/query_diag/`、mmdet3d 库。
- 第 7.4 节需要在 `tools/query_diag/` 中加的"读取实际初始位置"功能，按计划等第 0 步通过后、第 1 步之前再加。
- 第 1 步的训练和测试入口（以及 S-T3、S-T4）也在第 0 步通过后再写。

## 1. 文件

```
projects/QueryLoc/
  queryloc/__init__.py     注册（custom_imports 入口）
  queryloc/model.py        QueryLocBEVFusion：冻结 + train() 覆盖
  queryloc/head.py         QueryLocTransFusionHead：连续初始位置（dark / offset）、
                           双线性取特征、M-a 目标、M-b 偏移头与稠密 L1、
                           L>1 的 get_targets 修复、D3 复制初始化
  queryloc/subcell.py      DARK / soft-argmax（0a 已用，未改）；新增 bilinear_gather
  configs/ql_protocol.py   第 3 节协议
  configs/ql_c0.py         C0：原版 head，只加协议和冻结
  configs/ql_d3.py         D3：3 层、辅助监督、子类 head（V1）
  configs/ql_mb.py         M-b
  configs/ql_ma.py         M-a
  configs/ql_0b.py         第 0b 步（只用于测试）
  tools/ql_common.py       服务器工具共用的函数
  tools/check_load.py      S-T0
  tools/rehearse.py        第 4.2 节预演 + S-T2
  tools/eval_ql.py         单卡 val 评测（S-M0、0b），输出 oracle 格式的 json
  tools/pair_check.py      S-M1（第 0 步）；S-T5（第 2 步用）
  tools/run_queryloc.sh    新增 check-load / sm0 / sm1 / 0b / rehearse 五个步骤
  code_md5.json            19 个文件的登记表（S0''' 据此核对）
projects/DomeDETR/tests/
  ql_stub_env.py           CPU 测试环境：加载真实的 transfusion_head.py 等文件，
                           mmdet / mmcv 的依赖用替身
  test_exec3_model_cpu.py  模型部分的 CPU 测试（9 项）
  test_exec3_cpu.py        扩展了运行器测试
VERSION                    2026-10-22 一行（服务器侧版本记录）
```

## 2. v1.1 各项的落实

| 项 | 实现 | 验证 |
| --- | --- | --- |
| V1 / P-2：D3 用子类 head | `QueryLocTransFusionHead.get_targets` 按样本切 `preds_dict[0]`。C0 用原版 `TransFusionHead`（`ql_c0.py` 只有协议和冻结） | L = 1 时目标和损失与原版逐元素相同；3 层时逐层目标与单层原版相同（S-D3L，见第 4 节） |
| V2：协议写死 | `ql_protocol.py`：<br>• AdamW lr 2e-5（betas 取默认 (0.9, 0.999)，不加 momentum 调度）、wd 0.01、AMP 512、累积 4、裁剪 10；<br>• LinearLR 起始系数 1/3、500 iter；cosine 在 N 个 epoch 内退火到 0；<br>• `sync_bn='torch'`；`val_interval = max_epochs = N`，checkpoint 也只在第 N 个 epoch 保存（`max_keep_ckpts=1`）；<br>• `seed=0`，`deterministic=False`。<br>卡数：`run_queryloc.sh rehearse` 要求恰好 4 张卡，每张都通过 C5 检查 | 配置在 mmengine 下解析结果已核对。实际的 lr、betas、有效 batch 由预演写入报告（S-T3 第 1 步再查） |
| V3 / P-6、P-8：M-b | δ 在格子上取出后 detach，再加到 `query_pos`。L1 = \|Δx − t_x\| + \|Δy − t_y\|，对批内所有监督格求和后除以监督格数，λ = 0.25。偏移头在 fp32 下计算（P-7），lr_mult 10（P-9），最后一层零初始化 | 手算目标（两个 GT 的 7×7 窗口重叠，按最近 GT 归属，49 + 21 个监督格）；检测损失的梯度不进入偏移头 |
| V4 / P-11：DARK 退回条件 | `subcell.dark_offsets`：只有 `h_rr < 0` 且 `det > 0`（负定）并且不在边界格时才用 DARK，否则退回格子中心。0a 已用这一实现 | 平坦、鞍点、边界三种退回都有测试 |
| V5：S-M1 | CPU 测试，加上服务器端的 `pair_check.py sm1`：同一个训练 batch，C0 与零偏移 M-b 从同一份 backbone 特征出发，用同一个随机种子，比较 AMP 和 fp32 两种精度下除 `loss_offset` 之外的每一项损失 | CPU：`layer_-1_loss_bbox`、`layer_-1_loss_cls`、`loss_heatmap`、`matched_ious` 完全相同 |
| V5：S-D3L | CPU 测试 | 见第 4 节 |
| P-3：D3 复制初始化 | 在 head 上注册加载前置 hook：state_dict 有第 0 层、缺第 i ≥ 1 层时，用第 0 层补齐（包括 BN 缓冲区） | mmengine 和 torch 两种加载器都测过，并按 runner 的顺序先 `init_weights` 再加载；已有 3 层的 state_dict 不会被覆盖 |
| P-4：冻结 | `QueryLocBEVFusion(frozen_modules=...)`：构造时 `requires_grad_(False)`；覆盖 `train(mode)`，每次都把冻结模块设回 eval | 经过 `train()` 和类似 SyncBN 的模块替换后仍保持；可训练集合恰好是 fuser、BEV backbone / neck 和 head |
| P-5：双线性取特征 | `bilinear_gather`：4 次 gather 加权，位置为 (x − 0.5, y − 0.5) | 格子中心处与原 gather 逐位相同（fp32 和 fp16） |
| P-10：M-a 目标 | 照搬原实现的网格计算、半径、σ 和 eps 规则；GT 所在格为 1，其余格以连续中心计算高斯值 | GT 在格子中心时与原目标的差 ≤ 1e-6（原实现中 GT 网格坐标有 float32 舍入），正样本完全相同；GT 偏离格心时，高斯向真实中心偏斜 |
| P-12：top-K、标签、分数不变 | 只从原逻辑取 | S-T5（CPU）：M-a（DARK）和 M-b（非零偏移）选出的格、`query_labels`、`query_heatmap_score` 与原版完全相同 |

**第 0b 步（`ql_0b.py`）**：`query_init='dark'`、`feat_sampling='cell'`。初始 `query_pos` 用 DARK 位置，特征仍取峰值格，与 v1.1 第 5.2 节一致。

## 3. 服务器步骤

都通过 `run_queryloc.sh` 运行，每一步都先做 S0'''。

| 步骤 | 内容 | GPU | 预计耗时 | 输出 |
| --- | --- | --- | --- | --- |
| `check-load` | S-T0：四个配置分别与起点权重比对键；加载后核对 D3 的第 1、2 层等于第 0 层、M-b 的最后一层为零、参数划分符合第 3 节。<br>另报告训练集标注文件是否存在：#1、#2 在服务器上只读过 val，sm1 和预演要读训练集；这一项不计入 S-T0 | 无 | 几分钟 | `step0/st0/check_load.{txt,json}` |
| `sm1` | S-M1（1 个训练 batch） | 1 | 几分钟 | `step0/sm1/sm1.{txt,json}` |
| `sm0` | S-M0：先做 50 帧逐位比对（双线性取出的 query 特征和位置与原 gather 相同），再做 val 全量，\|ΔmAP\|、\|ΔNDS\| 相对 E0 都 ≤ 0.001 | 1 | 约 35 分钟 | `step0/sm0/eval_sm0_mb_zero.{txt,json}` 和 oracle json |
| `0b` | 第 0b 步：val 全量，输出相对 E0 的 ΔmAP、ΔNDS 和 breakdown | 1 | 约 30 分钟 | `step0b/eval_p0b_dark.{txt,json}` 和 oracle json |
| `rehearse all` | 依次对 C0、D3、M-b、M-a 用真实训练循环（4 卡 DDP、SyncBN、AMP、累积）跑 100 iter，然后停止（不保存、不验证）。记录：<br>• s/iter（去掉前 20 iter）、各卡峰值显存、每 epoch 的 iter 数和推算的墙钟时间；<br>• 参数划分、优化器各组的 lr 和 betas、有效 batch。<br>S-T2：冻结模块的参数和缓冲区前后逐张量相同；训练中冻结模块处于 eval、可训练模块处于 train；可训练参数确实被更新；损失没有 NaN / Inf；起始状态与 S-T0 相同（在 `init_weights` 和 `load_from` 之后读取，即真实训练路径上 M-b 的零初始化和 D3 的复制） | 恰好 4 | 每个配置约 5–8 分钟 | `step0/rehearse/ql_<名>/rehearse_ql_<名>.{txt,json}` |

**失败处理：**
- 任何一步失败，运行器以退出码 4 停止，并打印"Stop and report"。
- S-M0 或 S-M1 失败，M-a 和 M-b 都不能开跑；S-D3L 未通过，D3 不能开跑。

## 4. S-D3L：按字面判定不成立，需要你裁定

**v1.1 的判据**："3 层时，每一层的分类损失和回归损失，与'把该层预测单独送入原版 1 层的 `get_targets` 和损失函数'得到的值逐项相同。"

**事实（审计第 2 条已更正）。** 原版 `loss_by_feat` 对每一层都用 `avg_factor = 所有层的正样本总数`（`transfusion_head.py:550`、`:818`、`:864`），TransFusion 原仓库也是这样。因此在原代码下：

> 每层损失 = 该层单独计算的值 × n_该层 / n_总

这个等式是精确的。只要层数大于 1，字面上的"逐项相同"就不可能成立。

**CPU 测试结果**（`results/exec03/model_cpu_tests.txt`）：
- 三层的 labels、label weights、bbox targets、bbox weights 和 heatmap 目标，都与单层原版逐元素相同；
- 六项损失（3 层 × cls / bbox）与单层计算之比都恰好是 12/36（每层 12 个正样本，共 36 个）。

| 选项 | 内容 | 影响 |
| --- | --- | --- |
| **A（建议）** | 保留原归一化。把 S-D3L 的判据改为："逐层目标逐元素相同；每层损失 = 单层值 × n_该层 / n_总（浮点容差内）"。现有测试就是按这个判据通过的 | 符合 V1"子类只修 IndexError"的范围。三层损失之和与 C0 的单层损失量级相同，共享部分（fuser、BEV backbone / neck、shared_conv）上 heatmap 损失与解码器损失的比例与 C0 一致。各层独有的参数用 AdamW 更新，对损失的整体缩放基本不敏感 |
| B | 改成每层用自己的正样本数归一化，使字面判据成立 | 解码器损失总量约变为 3 倍，改变了共享部分上 heatmap 损失与解码器损失的比例，D3 与 C0 的比较会混入这一变化；这也是对原训练目标的第二处改动，超出 V1 |

**A 的代价**：在 D3 中，最后一层的损失只占解码器损失的约 1/3。这是辅助监督本身的设计，不是新引入的；登记在 D3 的结果旁即可。

## 5. 开发侧验证

**CPU 测试：**
- `test_exec3_model_cpu.py` 9/9 PASS，报告在 `results/exec03/model_cpu_tests.txt`。
- `test_exec3_cpu.py` 7/7 PASS。运行器测试新覆盖：
  - 每个新步骤的完整参数（例如 sm0 必须带 `--tag sm0_mb_zero --tol 0.001 --debug-frames 50`）；
  - `rehearse all` 按 c0、d3、mb、ma 的顺序发出 4 次 `torchrun --nproc_per_node=4`，每次前都对 4 张卡做 C5 检查；
  - 只给 1 张卡时拒绝；名称错误时拒绝；工具失败时退出码为 4。
- 回归：`test_query_diag_cpu.py`（带 `--ref-dir`）18/18、`test_exec2_cpu.py` 11/11、`test_exec2a_cpu.py` 6/6。
- 门禁：新文件全部通过非 ASCII、Python 3.8 语法和 `bash -n` 检查，flake8 无告警。

**CPU 测试中哪些是真实代码，哪些是替身：**
- 真实代码：`transfusion_head.py`、`transformer.py`、`utils.py`（HungarianAssigner3D、bbox coder）、`gaussian.py`，以及 QueryLoc 的全部代码。
- 替身：mmdet / mmcv 的组件，包括 ConvModule、损失函数、FocalLossCost、BboxOverlaps3D（BEV 轴对齐 IoU）、解码层的 attention / FFN、SeparateHead、BEVFusion 外壳。
- 因此真实的损失数值、AMP、SyncBN、DDP、真实数据和 checkpoint 加载，只能由服务器上的 S-T0、S-M1、S-M0 和预演来验证，CPU 测试替代不了。

## 6. 部署与运行

**部署**：同 #2a。新 zip 解压到新建的空目录，**只拷 `projects/QueryLoc/` 和 `VERSION`**。

```bash
SRC=/212022085500129/dome-detr/_src_exec03_step0
mkdir -p $SRC && unzip -q <新 zip 的路径> -d $SRC
NEW=$SRC/mmdetection3d-claude-dome-detr-query-diag
WS=/212022085500129/dome-detr/mmdetection3d-claude-dome-detr-query-diag
cp -r $NEW/projects/QueryLoc $WS/projects/
cp $NEW/VERSION $WS/VERSION
cd $WS && source tools/query_diag/exec02_env.sh
```

**现在就能跑的（不需要 4 卡）：**

```bash
bash projects/QueryLoc/tools/run_queryloc.sh check-load
nvidia-smi
CUDA_VISIBLE_DEVICES=0 bash projects/QueryLoc/tools/run_queryloc.sh sm1
CUDA_VISIBLE_DEVICES=0 bash projects/QueryLoc/tools/run_queryloc.sh sm0   # 约 35 分钟
CUDA_VISIBLE_DEVICES=0 bash projects/QueryLoc/tools/run_queryloc.sh 0b    # 约 30 分钟
```

**等 4 张卡都空闲后：**

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=0,1,2,3 bash projects/QueryLoc/tools/run_queryloc.sh rehearse all
```

`CUDA_VISIBLE_DEVICES` 里的卡号换成 `nvidia-smi` 中实际空闲的卡。C5 检查不通过时运行器直接拒绝，不会抢占。端口冲突时用 `PORT=<其他端口>` 覆盖默认的 29541。

**回传：**
- `work_dirs/queryloc/step0/`（st0、sm1、sm0、rehearse 下的 txt 和 json）
- `work_dirs/queryloc/step0b/`（txt、json 和 oracle json）
- `work_dirs/queryloc/s0ppp/s0ppp_check.txt`
- `work_dirs/queryloc/logs/` 下本轮的日志
- 可选：`work_dirs/queryloc/step0a/subcell_0a_per_gt.npz`，用于按类别拆分 0a 的召回

## 7. 已知风险

1. **训练集标注文件。** `check-load` 会报告它是否存在。如果缺失，sm1 和预演无法运行，需要先把数据目录补齐（不改代码）。
2. **S-M0 的全量容差。**
   - 0.1 pp 的容差只比同权重两次推理之间约 0.07 pp 的差（#2）大一点。
   - 真正检验实现的是 50 帧逐位比对。
   - 如果逐位比对通过、全量差值略超容差，报告仍判 FAIL，我会把两项数据一起交给你裁定，不自行重跑。
3. **GPU。** 预演需要 4 张同时空闲的卡。探针时 1–3 号卡被另一个容器占用，只能等待。
4. **预演时间。** 第 3 节的耗时是估计值。N 由预演实测的单 epoch 时间决定：超过 10 小时则 N = 1，所有运行用同一个 N。
