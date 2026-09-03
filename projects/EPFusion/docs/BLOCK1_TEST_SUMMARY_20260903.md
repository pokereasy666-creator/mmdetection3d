# EP-Fusion 块1测试总结与验收记录

记录日期：2026-09-03。结论范围：单卡、当前 checkpoint 与测试 batch 下的受控核心验收。

## 1. 最终结论

**EP-Fusion 块1单卡受控后端核心验收通过：10/10，进程退出码为 0。**

```text
selected_backend=no_cudnn
10/10 通过
sanity_exit=0
```

这项结论包含三种模式前向、日志键、Λ 初始化、单次反传、冻结参数梯度、两条 zero corruption 路径、scheduler 初始化与配置边界、R0 权重拷贝，以及受控教师路径和融合初始化等价性检查。

必须保留以下边界：

- `i_teacher_equivalence` 显式采用 `--equivalence-backend no_cudnn`，仅对两个 fuser 的受控复算禁用 cuDNN。阈值仍为 `rtol=1e-4, atol=1e-5`。
- 原始后端比较仍未满足该阈值，报告完整保留 `original_ok=False` 与 WARNING；不能将本结论写成“原始 cuDNN 后端逐元素验收通过”。
- 该选项不修改训练后端、模型算法、checkpoint 或上游代码。未传选项时，默认仍使用原始后端决定 `i` 是否通过。
- 初始化失败、运行异常和非有限值不能被受控结果掩盖；输入、原始输出及 Conv/BN 中间结果的非有限值也属于硬失败。
- 本次不是四卡 DDP 验收，也不是完整训练、收敛或检测指标验收。

## 2. 版本、执行条件与证据来源

| 项目 | 记录 |
| --- | --- |
| 仓库 | `pokereasy666-creator/mmdetection3d` |
| 当前工作分支 | `claude/EPFusion`，不再使用旧名 `claude/happy-bohr-21qiz2` |
| 对应本地验收脚本版本 | `ff0d89ae77775f1e9ed79395d8e8573407c76238` |
| 本地提交说明 | `Add explicit EPFusion equivalence backend selection` |
| 脚本标识 | `block1-explicit-equivalence-backend-20260903` |
| 执行位置 | 用户离线 Linux 服务器，`bevfusion` 环境 |
| 运行方式 | `CUDA_VISIBLE_DEVICES=0`，`--device cuda:0`，未使用分布式 launcher |
| batch 构建 | 脚本设置 `batch_size=2`、`num_workers=2`，取 dataloader 的一个 batch，并复用于该次检查 |
| 配置覆盖 | 命令使用 `randomness.seed=577127641` |
| 最终输出来源 | 用户提供的 2026-09-03 受控验收日志摘录，包含 10 项结果和 shell 退出码 |
| 发布状态 | 上述脚本提交仅在本地；按用户要求，未 push GitHub |

主配置：

```text
projects/EPFusion/configs/epfusion_m0_poe_4xa30-amp-accum_nus-3d.py
```

R0 配置：

```text
projects/EPFusion/configs/epfusion_m0_r0_convfuser_4xa30-amp-accum_nus-3d.py
```

用户提供并在复测命令中使用的真实 checkpoint 路径：

```text
/212022085500129/bevfusion-concat/mmdetection3d-claude-jolly-wozniak-c4YMQ/work_dirs/bevfusion_lidar-cam_official6e_4xa30_amp512_accum4_seed577127641/epoch_5.pth
```

最终服务器报告路径：

```text
/212022085500129/bevfusion-concat/mmdetection3d-claude-EPFusion/work_dirs/epfusion_block1_sanity_controlled/sanity_lambda_logging_20260903.txt
```

证据附件编号为 `b5b7606b-7b97-41dd-9025-9e4e392ca8bb/pasted-text.txt`，附件文件 SHA256 为：

```text
824120b59a031af14fd14f6ad5f719510df7081e43d7709d768933825d8c04c6
```

以上哈希标识的是本地收到的粘贴附件，不是服务器完整报告或 Python 脚本的哈希。最终附件未包含报告头的 `SCRIPT_SHA256`、完整命令行及环境版本；脚本版本、checkpoint 和命令参数按本次工作记录归档，尚未独立核对服务器 Git HEAD。

`M0_RECON_REPORT.md` 的历史探针记录为 PyTorch 2.0.1+cu118、CUDA 11.8、cuDNN 8700、MMEngine 0.10.5、MMCV 2.1.0、MMDetection 3.2.0、Python 3.8.20。这是历史环境信息，不应写成最终附件重新证明的环境快照；同样，配置文件名包含 `4xa30` 不代表此次测试使用了四卡。

## 3. 测试推进与问题闭环

| 阶段 | 主要现象 | 处理与结论 |
| --- | --- | --- |
| 前置脚本补丁 | `--cfg-options` / `--data-root` 未应用；zero-path 缺键可能仍 PASS；随机张量存在 device 隐患 | 配置覆盖、递归 data-root patch、真实缺键断言与 CPU 随机结果的 device 转移得到补充 |
| 服务器路径检查 | `work_dirs -> work_dirs` 自引用，出现 `Too many levels of symbolic links` | 定位为部署路径问题；使用真实 checkpoint 路径，后续运行已能写入报告。本记录不推断最终软链接修复方式 |
| 首轮运行 | `g_zero_paths_smoke` 出现 CUDA OOM；`d`、`i` 未通过 | 冒烟前向禁用梯度、日志保存为 detached CPU 数据；反传使用独立前向并清理梯度；增加受控精度和逐阶段诊断 |
| 内存与路径诊断 | 8/10；zero-path 通过；stock 自己重复前向首先在 depth 处不一致 | 差异与上游 depth 索引覆盖写入的非确定性一致。没有修改上游 depth 算法，改在等价性检查中控制确定性 |
| 确定性与 FP64 诊断 | 9/10；`d` 各阶段完全一致；`i` 仍有少量超差 | `d` 问题在受控条件下消失；局部 FP64 参考将 `i` 的主要差异定位到 teacher FP32 路径 |
| Conv/BN 与后端交叉检查 | 原始 `i` 仍 FAIL；仅禁用 fuser 的 cuDNN 后，全输出按原阈值通过 | 误差主要来自 teacher Conv，再经 BN 缩放；BN 自身误差很小。此时后端结果仍是诊断项，因此总结果保持 9/10 |
| 显式选择验收后端 | 用户确认采用 `--equivalence-backend no_cudnn` | 默认行为保留；显式选择时以受控结果为 `i` 闸门，同时完整记录原始比较及 WARNING；最终 10/10、exit 0 |

期间还处理了两项脚本级问题：scheduler 构造后不再额外手动 `step()`，改为读取初始化状态；新输出目录必须先创建，否则 `Tee` 打开报告文件会报 `FileNotFoundError`。

后续运行未再出现该次 OOM，不等于已经证明训练显存峰值或长时间稳定性。各轮 batch 未证明完全一致，不能把跨轮的 loss 或最大误差差值直接解释为定量改进幅度。

## 4. 最终十项检查结果

| 验收项 | 最终结果 | 日志证据 / 覆盖内容 |
| --- | --- | --- |
| `i_teacher_equivalence` | PASS（受控后端） | `original_finite=True`，`no_cudnn_ok=True`，全输出 `bad=0/16588800` |
| `c_no_object_sample` | PASS | 展开的训练 pipeline 中没有 `ObjectSample` |
| `run_three_modes` | PASS | clean、corrupt_cam、corrupt_lidar 均完成 loss 与日志解析；每种模式 `log_vars=23` |
| `a_keys_present` | PASS | `missing=none`，所需 Λ、teacher 和预警日志键在场 |
| `b_lambda_init_one` | PASS | `mean(Lam_C)=1.000000`，`mean(Lam_L)=1.000000` |
| `f_backward_smoke` | PASS | `train_no_grad=[]`，`frozen_has_grad=[]` |
| `g_zero_paths_smoke` | PASS | `corrupt_cam/zero_image` 与 `corrupt_lidar/zero_points` 均 `missing=none` |
| `h_optim_scheduler` | PASS | 初始化值可读取，`max_epochs=3`，`out_of_bounds=none` |
| `d_teacher_allclose` | PASS | 教师状态一致；stock 自重复与 EP/stock 逐阶段差值均为 0 |
| `e_r0_weight_copy` | PASS | `fusion_mode='convfuser'`、`w_teach=0.0`、`emit_clean=False`、`copied=True`、`all_equal=True` |

最终三种模式的 loss_sum 分别为：clean `2.71534`、corrupt_cam `2.71466`、corrupt_lidar `2.68807`。这些是同次冒烟前向的输出，不是训练曲线或鲁棒性指标。

scheduler 初始化报告为：

```text
base_lrs=[2e-05, 0.0002]
lr_unique=[6.6666666e-06, 6.6666666e-05]
momentum_unique=[0.9]
groups=135
max_epochs=3
out_of_bounds=none
```

这只验证初始化和配置边界，不证明经过梯度累积后的实际 optimizer/scheduler 更新时序。

## 5. 教师路径与数值误差分析

### 5.1 `d`：受控教师路径一致

检查设置为 `strict_fp32=True`、`deterministic=True`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`，并得到 `teacher_state mismatches=none`。

两组比较——`stock_self_repeat` 和 `ep_vs_stock`——在以下全部位置均为 `max_abs_diff=0`、`bad=0`：

- 输入图像、两份点云；
- camera2lidar、camera_intrinsics、img_aug_matrix、lidar2image、lidar_aug_matrix；
- `img_feats_2d`、`depth`、`camera_bev`、`lidar_bev`、`fused`。

两组均为 `first_difference=none`。该结果证明当前 batch、checkpoint 和受控设置下的路径一致，不能外推为所有输入或所有运行设置逐位一致。

### 5.2 `i`：全输出比较与原始后端审计

| 比较 | 是否通过原阈值 | 最大绝对误差 | 超差元素 |
| --- | --- | --- | --- |
| 原始 PoE vs teacher | 否；保留 WARNING | `4.693e-04` | `872/16588800` |
| cuDNN-disabled PoE vs teacher | 是；正式受控闸门 | `7.248e-05` | `0/16588800` |
| cuDNN-disabled teacher vs 原始 teacher | 否；后端诊断 | `4.698e-04` | `877/16588800` |
| cuDNN-disabled PoE vs 原始 PoE | 是；后端诊断 | `4.005e-05` | `0/16588800` |

上述全输出比较均报告 `finite=True`。容差按每个元素的 `atol + rtol * abs(reference)` 判断，所以最大绝对误差大于 `atol` 并不自动意味着失败。

### 5.3 FP64 分解：覆盖全部 872 个原始超差坐标

下表仅统计原始超差坐标，与上一表的全输出统计范围不同。两个后端都在相同的这组坐标上诊断；FP64 参考不会替换真实验收输出。

| 分解项 | 原始后端 | cuDNN-disabled |
| --- | --- | --- |
| `teacher_fp32_vs_ref64` | `4.698e-04` | `2.104e-06` |
| `poe_fp32_vs_stored64` | `2.806e-06` | `2.250e-06` |
| `stored64_vs_truefold64` | `2.390e-07` | `2.390e-07` |
| `truefold64_vs_expected64` | `4.996e-16` | `4.996e-16` |
| `expected64_vs_teacher64` | `1.215e-06` | `1.215e-06` |
| `conv_fp32_vs_ref64` | `1.267e-04` | `5.236e-07` |
| `conv_error_after_bn_scale` | `4.697e-04` | `2.105e-06` |
| `bn_fp32_vs_same_input64` | `2.013e-07` | `1.392e-07` |
| `bn_fp32_vs_ref64` | `4.698e-04` | `2.104e-06` |

在此批输入上，证据支持以下解释：

1. 原始 teacher 卷积的数值偏差经 BN 缩放后形成约 `4.698e-04` 的输出偏差；BN 在相同 Conv 输入上的自身计算误差只有约 `2e-07`。
2. 禁用两个 fuser 的 cuDNN 后，teacher 相对 FP64 的偏差降至约 `2e-06`，且 PoE/teacher 全输出按原阈值通过。
3. PoE 存储权重与真正 FP64 折叠结果差异很小，FP64 折叠与理论期望的差异接近双精度舍入量级。
4. Λ=1、当前 ReLU 设置下，理想 PoE 输出因分母 `eps` 存在 `2/(2+eps)` 因子；该影响已单独记录，并非本次原始后端大误差的主因。

这定位到了当前后端配置下的计算路径差异，不等于证明 cuDNN 软件 bug，也未定位到某个具体 kernel。无需据此修改 PoE 公式、折叠初始化、损坏分布或 `eps`。

## 6. 代码变更与本地验证记录

本轮服务器排障后的以下提交均只修改 `projects/EPFusion/scripts/sanity_lambda_logging.py`：

| 提交 | 用途 |
| --- | --- |
| `1966a6bf` | 清理冒烟计算图和日志引用；加入 strict FP32、教师状态与逐阶段诊断 |
| `204a4104` | `d` 受控确定性、`i` 局部 FP64 分解、scheduler 初始化预览修正 |
| `fc45c996` | 只读 Conv/BN hook、同特征的 cuDNN-disabled fuser 对照 |
| `ff0d89ae` | 显式验收后端选项，保留原始审计，强化非有限值与异常处理 |

`fc45c996` 与 `ff0d89ae` 按用户要求只提交到本地，未上传 GitHub。本表不将更早的前置配置或初始化补丁混同为“仅改脚本”。

最终代码补丁的本地验证记录：

```bash
python -m py_compile \
  projects/EPFusion/scripts/sanity_lambda_logging.py \
  projects/EPFusion/epfusion/corruptions.py \
  projects/EPFusion/epfusion/preprocessor.py \
  projects/EPFusion/epfusion/ep_fusion.py \
  projects/EPFusion/epfusion/poe_fuser.py \
  projects/EPFusion/epfusion/hooks.py
```

结果：六文件通过，编译器无输出；`git diff --check` 通过。显式后端补丁另有 12 项临时 AST 抽取/小模型测试通过，运行环境为本地 `torch 2.13.0+cpu`、无 CUDA。覆盖 CLI 默认值与非法值、默认模式不自动放行、显式模式的双向通过/失败组合、退出码、初始化失败、原始/受控路径异常、NaN/Inf、被 ReLU 隐藏的非有限中间值，以及状态恢复。

这些是本地隔离测试，不是完整项目集成测试，也不是服务器 CUDA 执行结果；真实服务器证据来自本报告第 4、5 节。

## 7. 复测命令与归档要求

将对应版本的本地脚本复制到服务器同路径后，在服务器仓库根目录执行：

```bash
mkdir -p work_dirs/epfusion_block1_sanity_controlled && \
CUDA_VISIBLE_DEVICES=0 \
python projects/EPFusion/scripts/sanity_lambda_logging.py \
  --config projects/EPFusion/configs/epfusion_m0_poe_4xa30-amp-accum_nus-3d.py \
  --r0-config projects/EPFusion/configs/epfusion_m0_r0_convfuser_4xa30-amp-accum_nus-3d.py \
  --checkpoint /212022085500129/bevfusion-concat/mmdetection3d-claude-jolly-wozniak-c4YMQ/work_dirs/bevfusion_lidar-cam_official6e_4xa30_amp512_accum4_seed577127641/epoch_5.pth \
  --device cuda:0 \
  --out-dir work_dirs/epfusion_block1_sanity_controlled \
  --equivalence-backend no_cudnn \
  --cfg-options randomness.seed=577127641
rc=$?
echo "sanity_exit=$rc"
```

- 脚本以日期命名报告，同一天复用同一输出目录会覆盖旧文件；重新运行前应归档旧报告或另选 `--out-dir`。
- `randomness.seed` 已作为 config 覆盖值传入，但脚本手动构建 dataloader，未显式建立完整的全局播种流程。因此该参数不构成跨运行 batch 或结果完全一致的保证。
- 严格追溯时应另行保存完整报告头、`SCRIPT_SHA256`、服务器代码版本、环境快照、checkpoint 标识和实际样本标识；这些额外元数据不在当前最终摘录中。
- `--equivalence-backend` 是 sanity 专用参数，不应直接传给正式训练脚本。

## 8. 下一阶段与未验收内容

块1此次单卡受控验收可以关闭。下一阶段应独立进行四卡短程运行验收，至少检查：

1. 四个 rank 正确初始化、数据分片与同步正常，无 DDP unused-parameter 或通信错误。
2. 多卡反传正常，每个 rank 的冻结/可训练参数状态正确。
3. 多个真实 optimizer 更新、梯度累积边界及不足完整累积窗口的末尾行为符合预期。
4. scheduler 的实际更新时序、学习率与动量行为正确。
5. Λ 日志的跨 rank sum/count 聚合正确，无错误的 rank 均值替代。
6. 训练显存峰值、AMP 数值和短程稳定性满足要求，再决定是否进入完整训练。

目前没有四卡运行、完整 epoch、收敛、mAP/NDS 或损坏鲁棒性指标的验收证据，不能在本总结中将它们标为通过。

推荐后续统一引用的结论：

> EP-Fusion 块1于 2026-09-03 完成单卡受控后端核心验收，10/10 通过，进程退出码为 0。融合初始化等价性显式采用 cuDNN-disabled fuser 复算，保持原容差；原始 cuDNN 后端的数值差异完整保留为审计 WARNING。该结论不代表原始后端逐元素通过，也不代表四卡训练或性能指标已验收。
