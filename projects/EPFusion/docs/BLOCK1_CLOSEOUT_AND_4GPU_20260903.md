# EP-Fusion 块1收尾审计与四卡交接

日期：2026-09-03。依据用户《EPFusion_块1收尾与四卡推进_20260903.md》、本地代码及已提供的服务器日志。

## 1. 当前结论与证据边界

单卡受控 sanity 已 10/10、exit 0；**块1整体仍未关闭**。这轮完成代码核验、证据采集补丁、发布与可追溯部署材料，不伪造服务器执行结果。

- 旧 `work_dirs -> work_dirs` 曾自引用，但成功写出最终报告只证明该次输出路径可用。当前链接目标及 config 相对 `load_from` 是否可达，须在服务器重新验证；不能断言“仍未修”或“已经修好”。
- `fc45c996` / `ff0d89ae` 有真实本地提交；缺的是服务器文件哈希和 GitHub 发布映射，不是“文件状态不在任何仓库”。
- ZIP 默认无 `.git`，服务器不能调用 `git rev-parse HEAD`；版本链使用固定 GitHub SHA 下载地址、ZIP SHA256、文件校验清单、VERSION 与报告头。
- 现有最终摘录没有 F_T 量级和三模式 teacher 明细；旧脚本也未打印这些数值。新增输出须由重跑获取，不能从 `loss_sum≈2.7` 倒推，更不能据此直接决定 R2。
- 原始 i 检查运行于 strict FP32（关闭 TF32/AMP），不是生产 AMP 前向。其误差不能直接写成生产 AMP 误差，更没有证据确认使用了 Winograd 等具体 kernel。

## 2. 前置补丁审计：实现存在，运行证据有缺口

原文称“12 项”，但表内 A1–A4 + B1–B9 为 13 个标签；下表按标签逐项核验，不按预期猜缺失。

关键历史提交：

- 第一轮：`a23577bf13a921b3cd4ee00b96ebad27680ba49f`，参数覆盖、zero-path 真断言、随机张量 device。
- 第二轮：`a93cfe7a49ac2c08862d65b9cb027f0662fd133a`，`EP-Fusion block1 patch2: teacher init and sanity hardening`。
- 第二轮 VERSION 标记为 2026-07-26；Git 提交记录日期为 2026-08-30。标签日期不等同于提交日期。

以下路径均相对 `projects/EPFusion/`，除明确标出的根 VERSION。

| 标签 | 核验位置 | 实现与证据 |
| --- | --- | --- |
| A1 | `epfusion/poe_fuser.py`：构造、`init_from_convfuser`、`forward` | 3×3、折 BN、通道切分×2、输出 ReLU 均已存在；不重写 |
| A2 | `epfusion/hooks.py`：`PoETeacherInitHook.before_train` | iter=0 初始化及 resume 分支存在；尚无真实 Runner hook 日志 |
| A3 | `epfusion/ep_fusion.py`：`_eval_student_fuser_norm`、`train` | `_BatchNorm.eval()` 已存在，未冻结 affine；本轮增加 sanity 运行断言 |
| A4 | 两份 `configs/epfusion_m0_*.py` | init/copy hook、退火默认关闭、load_from、proj_kernel=3、out_act=relu 均存在；路径实机待查 |
| B1 | `scripts/sanity_lambda_logging.py`：`_b` | 源码确为 `abs(mean-1)<1e-6`，不是仅凭六位小数推断 |
| B2 | 同脚本 `_h` | epoch 端点和 momentum 初始化预览存在；实际 optimizer/scheduler 时序仍待训练 |
| B3 | 同脚本 `run_mode` | 三模式均执行 parse_losses；本轮补齐 teacher/bbox 明细 |
| B4 | 同脚本 `detect_training_processes`、`main` | 进程自检已存在，但原先仅 WARNING 后继续；本轮改为无法确认空闲即退出 2 |
| B5 | 同脚本 `_i` | 初始化等价性与显式后端选择已存在；不改阈值和原始审计 |
| B6 | 同脚本 `_e`、`apply_data_root` | R0 配置断言及 dataloader/evaluator dict/list 递归覆盖均存在 |
| B7 | `epfusion/preprocessor.py`：clean 图归一化 | `data_samples=None` 时第二参数已为 `False`，不重复修改 |
| B8 | `docs/M0_PLAN.md`：B/F/G/I | epoch_5、0.7060/0.6648、R0 BN 决议已存在；本轮修正逐位等价与运行顺序旧表述 |
| B9 | 根 `VERSION` | patch2 标识存在；本轮按明确授权更新为 archival-evidence-v1 |

本轮不修改 ep_fusion、poe_fuser、preprocessor、corruptions、hooks 的算法或训练行为；不修改 `projects/BEVFusion/` / `mmdet3d/` / `scripts/deploy.sh`。根 VERSION 是用户此次文件指令明确授权的范围例外。

## 3. 此前本地提交的发布映射

原生 Git push 在本机没有可用凭据；通过 GitHub 连接器按相同 tree 发布。连接器生成的 author/committer 元数据不同，故 commit SHA 不同；每个完整 Git tree 已逐一比对相等。未 force push。

| 原本地 SHA | GitHub 发布 SHA | 相同的 tree SHA |
| --- | --- | --- |
| `fc45c9960dffeae5f46525c385c2ef98ceadfd22` | `3dc75d4e4ddc33041cd555f9316addab0fd60a6f` | `aa2ec660fb2a10dc28af37f7ca6b94699829d4c1` |
| `ff0d89ae77775f1e9ed79395d8e8573407c76238` | `c3d4ec88eae1e7ef227d9417e951614eb829121c` | `f08a1008f9e6007235cdae490d0c1aba861496de` |
| `4fb5c7d1a951578a484898b15d633881b8eb9632` | `8e1b9fb97964055288da1df40b25b37ef159c8ae` | `823698a9b9542fa46053498e10c8d7113db54b4f` |

本地保留 `backup/epfusion-local-before-publish-20260903`。工作分支已基于远端相同补丁链继续；本轮证据补丁的最终发布 SHA 见随包交付记录。VERSION 用唯一发布标签而非嵌入自身 commit SHA，避免自引用。

## 4. 新报告会采集什么

- 报告头：VERSION、SCRIPT_SHA256、调用方声明的 SOURCE_SHA、预期脚本哈希校验、完整参数、checkpoint 解析路径、主 config 的 load_from/custom_hooks/poe_cfg。
- i：原始 strict-FP32/cuDNN F_T 的 `F_T_abs_mean` / `F_T_abs_max`；保留原始/受控全输出比较与 FP64 诊断，不改 `rtol=1e-4, atol=1e-5`。
- 三模式：`loss_teach`（当前权重后）、`teach_nll_raw`、`bbox_loss_sum`、`teacher_to_bbox`、`raw_to_bbox`、bbox 键列表及 w_teach。分母非正时标为 undefined，非有限损失则失败。
- e：调用 `r0.train()` 后报告 `bn_count`、`bn_training`、`bn_frozen_affine`。BN 缺失、处于 train 或 affine 被冻结均失败。
- 非存在 checkpoint 不再退回随机权重；省略 CLI checkpoint 时使用 EP config 的 load_from。stock 对照继续不接收 EP 专属 cfg-options。
- 新目录自动创建，报告带时分秒和微秒，避免同日覆盖。训练进程/GPU compute 占用或扫描失败时，GPU 前向之前退出 2；不增加绕过选项。

`SOURCE_SHA` 是用户输入，不是哈希自证。必须先通过外部 ZIP/文件清单校验，再结合报告中的 SCRIPT_SHA256；不能只填一个 SHA 就宣称已验证版本。

### i 项的准确结论

已有 FP64 证据在原始超差的 872 个坐标上支持折叠实现与理论公式一致（`stored64_vs_truefold64=2.390e-07`、`truefold64_vs_expected64=4.996e-16`）；这不是对任意输入的机器证明。理想输出含 `2/(2+eps)` 因子。

原始 strict-FP32/cuDNN 全输出比较仍有 `872/16588800≈0.0053%` 元素超差，max abs diff `4.693e-04`；no_cudnn 同阈值全输出通过。F_T 量级、生产 AMP 行为与训练影响仍待记录，暂不将原始偏差判为“可接受”。

## 5. 按 SHA 部署：在离线服务器执行

这一阶段本地无法代执行。不得覆盖正在运行的旧目录。先检查 `ps -eo pid,args` 与 `nvidia-smi`；若有训练运行或状态未知，暂停部署和 GPU 测试，不杀进程、不更换其代码。

将交付的固定 SHA ZIP 和同 SHA 的 `.sha256` 清单一次上传。按交付记录核对 ZIP SHA256，再解压到全新目录。设置以下两个来自交付记录的值：

```bash
SOURCE_SHA='<最终发布的40位SHA>'
SCRIPT_SHA256='<交付记录中的64位脚本SHA256>'
# 在新解压的 mmdetection3d-$SOURCE_SHA 根目录运行：
sha256sum -c ../epfusion-$SOURCE_SHA.sha256 || exit 1
head -1 VERSION
```

清单使用 ZIP 内原始文件字节，不能拿 Windows 工作树 CRLF 文件的哈希替代。无需 `git`。清单、ZIP、下载 URL 和发布 SHA 一同保留。

部署路径必须先解析为存在的绝对路径，禁止 `--work-dirs work_dirs`：

```bash
OLD=/212022085500129/bevfusion-concat/mmdetection3d-claude-jolly-wozniak-c4YMQ
CKPT_REL=bevfusion_lidar-cam_official6e_4xa30_amp512_accum4_seed577127641/epoch_5.pth
WORK_REAL=$(realpath -e "$OLD/work_dirs") || exit 1
DATA_REAL=$(realpath -e "$OLD/data/nuscenes") || exit 1
test -d "$WORK_REAL" && test -d "$DATA_REAL" || exit 1
test -r "$WORK_REAL/$CKPT_REL" || exit 1
# OLD 是用户已给的 checkpoint 来源；data/nuscenes 与 CUDA ops 是否也在此需实机确认。
# 若不存在，停止，并将 --data/--ops-from 改为已核实的真实目录。
test "$WORK_REAL" != "$(pwd -P)/work_dirs" || exit 1
bash scripts/deploy.sh --data "$DATA_REAL" --work-dirs "$WORK_REAL" --ops-from "$OLD" || exit 1
ls -ld -- work_dirs
readlink -f -- work_dirs
ls -lh -- "work_dirs/$CKPT_REL"
test -r "work_dirs/$CKPT_REL" || exit 1
```

`deploy.sh` 会替换软链，但拒绝覆盖已存在的真实 work_dirs 目录。若拒绝，停止核对，**不要 rm -rf**。本轮没有更改 deploy.sh；其末尾旧 baseline1 示例不作为 EP 的 checkpoint 来源。

## 6. 归档式 sanity 重跑

先确认第 5 节路径与清单全部通过、GPU 空闲。在同一 shell 中保留 SOURCE_SHA、SCRIPT_SHA256、CKPT_REL：

```bash
OUT="work_dirs/epfusion_block1_sanity_archival_${SOURCE_SHA:0:12}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT" || exit 1
CUDA_VISIBLE_DEVICES=0 python projects/EPFusion/scripts/sanity_lambda_logging.py \
  --config projects/EPFusion/configs/epfusion_m0_poe_4xa30-amp-accum_nus-3d.py \
  --r0-config projects/EPFusion/configs/epfusion_m0_r0_convfuser_4xa30-amp-accum_nus-3d.py \
  --checkpoint "work_dirs/$CKPT_REL" \
  --device cuda:0 --out-dir "$OUT" \
  --source-sha "$SOURCE_SHA" --expected-script-sha256 "$SCRIPT_SHA256" \
  --equivalence-backend no_cudnn \
  --cfg-options randomness.seed=577127641
rc=$?
echo "sanity_exit=$rc"
```

保留完整报告（含头部），而不只是末尾十行。此命令未向主/R0 同时覆盖 model.w_teach，避免破坏 R0 的 w_teach=0 断言；EP 默认 1、R0 默认 0。

这里的 seed 仍是 config 覆盖；当前手动 dataloader 未显式传入 seed/全局播种，因此不保证跨次 batch 相同。本轮不改随机采样逻辑。文件可追溯与逐次结果完全相同是两件事。

## 7. 四卡与第一个 EP 正式 run 合并（暂未启动）

前置：归档报告、哈希、相对 checkpoint、R0 BN 断言均通过，teacher/bbox 数值已审阅。用三种模式的 `raw_to_bbox` 估计 R2(w=1) 的相对量级，再作选择：同量级可选 R2；明显超过约 10 倍时先考虑 R1(w=0.1)；中间区间/分母异常/模式差异大须复核，不自动猜测。负 NLL 也不能仅按带符号总 loss 判断。生产 AMP iter-0 再确认一次。

确认后从同一发布目录启动（不传 sanity 专属参数，不 resume）：

```bash
W_TEACH='<根据归档结果确认的0.1或1.0>'
case "$W_TEACH" in 0.1|1.0) ;; *) echo '先确认 R1/R2'; exit 1 ;; esac
RUN="work_dirs/epfusion_ep_w${W_TEACH}_${SOURCE_SHA:0:12}_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  tools/train.py projects/EPFusion/configs/epfusion_m0_poe_4xa30-amp-accum_nus-3d.py \
  --launcher pytorch --amp --sync_bn torch --work-dir "$RUN" \
  --cfg-options model.w_teach="$W_TEACH" randomness.seed=577127641 \
    model.data_preprocessor.seed=577127641 default_hooks.logger.interval=1
```

明确使用 4 rank、每卡 batch2、accum4、有效 batch32；不改 LR、不加 auto-scale-lr。logger.interval=1 用于首训观测，会增加日志量，不代表新增 optimizer 更新。

| 闸门 | 观察点 | 边界与动作 |
| --- | --- | --- |
| G1：启动后/首次 iter 前 | 4 rank、无 DDP/通信错误；`[PoETeacherInitHook] initialized poe_fuser from fusion_layer @iter0 (ok=True, ...)` | 这是现有真实日志格式，不要求原文中并不存在的连续字符串 `ok=True @iter0`；没有成功日志则停止 |
| G2：首批训练 | 冻结主干/教师 BN eval；loss_teach/bbox、显存峰值；R0 另查 student BN eval | PoE 没有 student_fuser；已有 sanity 不替代 Runner 后 model.train() 的运行证据。状态不明就暂停核验，不能自动打勾 |
| G3：前约 200 iter | 多个更新、无 NaN/Inf、AMP scaler 状态、耗时稳定、日志各键 | 无 AMP skip 且不在尾窗时，每 4 micro-iter 一个更新；不足完整窗口可能有末尾 step。不能把 iter/4 当作已观测 optimizer 次数，也不能忽略 scaler 跳步 |
| G4：epoch1 | cos_pc_ft/cos_pl_ft、按模式的 Λ sum/cnt 聚合、真实 clean val NDS | 作为复核信号而非单变量自动判坍缩；若 cnt=0 不算比值，不平均各 rank 的均值代替总 sum/总 cnt |

分钟数仅是观察窗口，不是机器速度保证。当前 Hook 不额外采集所有 rank 的 BN 模式或实际 AMP optimizer step 次数；这些项仍需真实 Runner 日志/运行诊断，不能声称普通十项 sanity 已覆盖。训练操作交由服务器用户执行，当前不自动启动或中止已有任务。

本轮不实现块2的 sanity_p1、sanity_p4、eval_quarter_val；epoch1 NDS 使用当前配置已有真实 val，不用尚未实现的脚本冒充结果。

## 8. 本地验证与关闭清单

六文件 `python -m py_compile`（sanity、corruptions、preprocessor、ep_fusion、poe_fuser、hooks）通过，无编译输出。

新增 `tests/test_block1_archival.py` 使用现有 pytest/torch 做 CPU AST 隔离测试，无新依赖、不导入 MMEngine/MMCV，不构造真实 Runner。18 项通过：量级只读、加权/原始比值区分、异常分母和非有限损失、BatchNorm/SyncBatchNorm train 后 eval 与 affine、进程检查、脚本哈希/来源 SHA、输出目录与防覆盖、missing checkpoint。

初次 pytest 因系统临时目录权限失败，切换到工作区新建临时目录后通过。不能将这些 CPU 测试记作 CUDA/四卡运行。

- [x] 此前本地提交内容已发布，原 SHA/发布 SHA/tree 映射归档。
- [x] A1–A4 / B1–B9 本地代码逐项核验；仅补证据与安全缺口。
- [x] 归档所需统计、BN 断言、哈希守卫与手册已提供。
- [ ] 新发布版本上传解压、ops/数据部署、相对 load_from 可达（服务器待执行）。
- [ ] 新归档式 sanity 的完整报告、哈希对应、10/10 和 exit0。
- [ ] F_T 量级与三模式 loss_teach/teach_nll_raw/bbox 数值归档，确认 R1/R2。
- [ ] 真实 Runner 初始化/copy hook、四卡 G1–G4 证据。

在上述服务器证据回传前，结论保持“sanity 已通过，块1收尾进行中；四卡未验收”。
