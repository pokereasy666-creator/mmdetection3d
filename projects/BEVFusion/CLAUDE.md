# EP-Fusion M0 试点 — 项目宪法

## 项目一句话
在冻结的 BEVFusion 主干上，给相机/LiDAR 分支各加"可信度头"(Λ)，
用闭式 PoE 加权融合替换 ConvFuser，通过教师对照 NLL + 退化演练训练 Λ 的校准。

## 环境事实
- GPU: 4 × A30 (24GB)。训练一律 torchrun 4 卡 + AMP fp16 + 梯度累积。
- nuScenes trainval 与 BEVFusion 官方 checkpoint 已就绪（路径见 configs/paths.py，首次会话先确认）。

## 不可违反的工程铁律（违反任何一条 = 实验作废）
1. 冻结模块（相机分支、LiDAR 分支、教师 ConvFuser）必须同时
   requires_grad_(False) 和 .eval()。只做前者会让 BN 统计量被损坏输入污染。
2. 教师特征 F_T 必须 .detach()。教师侧禁止添加任何可学习变换（投影只放学生侧）。
3. 增广顺序：几何增广（翻转/旋转/缩放）对两模态一致施加在前；
   退化损坏单模态施加在后（图像损坏作用于归一化前的原图，点云损坏作用于体素化前的原始点）。
4. M0 禁用 GT-sampling/GT-paste。
5. 损坏配置必须是两个分离的列表：TRAIN_CORRUPTIONS / TEST_CORRUPTIONS_DOC。
   训练代码禁止 import 或调用测试族（雾/雨/雪/运动模糊/眩光属于测试族，只写文档不写实现）。
6. Λ 参数化为 log-precision，clamp 到 [-7, 7]，末层零初始化（初始 Λ≈1）。
   PoE 分母加 eps=1e-6。
7. 干净样本同样计算教师对照 NLL（"小误差→高 Λ"是校准的一半）。
8. 每 N 步必须记录 clean / corrupt_cam / corrupt_lidar 三种模式下的
   mean(Λ_C) 与 mean(Λ_L)（这是活体 sanity 信号）。
9. 所有 run 固定 seed 并记录 git commit hash。
10. mmengine 特有：训练循环会反复调用 model.train()（epoch 开始、验证结束后）。
    冻结模块保持 eval 必须通过 override 模型类的 train(self, mode=True) 方法实现，
    在其中强制冻结子模块 .eval()。禁止只在构造函数里调一次 .eval()。
11. 上游隔离：禁止修改 projects/BEVFusion/ 与 mmdet3d 库内任何文件。
    全部新代码放在新建的 projects/EPFusion/ 下（模型类继承 BEVFusion 并注册，
    config 继承复现所用 config）。
12. Git 纪律：只在分支 claude/jolly-wozniak-c4YMQ 上工作；每完成一个模块
    （含其 sanity 通过）做一次独立 commit，message 注明模块名与验证结果。
13. 运行中训练保护：执行任何脚本前先检测是否有训练进程在跑（nvidia-smi / ps）。
    检测到正在运行的训练时：禁止任何 GPU 操作、禁止修改已有 tracked 文件、
    禁止 git checkout/切分支、禁止改动 conda 环境；只允许只读勘察和新建文件。
14. 探针工作流：需要 GPU/数据/训练日志才能回答的问题，一律打包成无副作用的
    探针脚本（输出汇总到单个文本报告），由用户在 GPU 服务器上执行后贴回结果，
    不得用估计值或文档默认值冒充实测值。

## M0 范围纪律（防 scope creep）
- 只做：逐格标量 Λ、教师对照(路线b)+退化演练(路线c)、PoE 融合。
- 不做（M1+ 再说）：NIG/evidential 头、协方差交(CI)、通道组 Λ、
  任务误差路线(a)、nuScenes-C 正式评测、注意力融合基线、DAL 骨干。

## 工作方式
- 重大改动先出计划，经确认后再写代码。
- 每个模块写完必须先跑对应的 sanity 脚本再继续下一个模块。
- 不确定 repo 结构时先勘察并提问，禁止凭记忆假设 mmdet3d 的 API。
