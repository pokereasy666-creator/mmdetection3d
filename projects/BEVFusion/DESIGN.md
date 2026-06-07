# DESIGN — 三模块开关化设计（GAFusion-SDG / DepthFusion-DGF / InsFusion）

> 范围：**仅设计**，不实现任何模块内部（本任务约束）。
> 公式/维度取自用户上传的三篇论文原文（用纯 stdlib 抽取）：
> - **GAFusion**（arXiv:2411.00340）/ 其扩展期刊版 **MGAF**（IEEE TPAMI 2026）**§III-B LiDAR Guidance**
> - **DepthFusion**（arXiv:2505.07398）**§III-B Depth-GFusion**
> - **InsFusion**（arXiv:2509.08374）**§3 Methodology**
>
> 标 **`⚠️ 仍需你定`** 的是论文无法回答、需你拍板的**集成决策**（mmdet3d 基线特有）。
> 实现阶段：每个重构组件的代码注释须引用论文**节号/公式号**（约定见 §8）。

---

## 0. 基线前向回顾（插入点坐标系）

文件均在 `projects/BEVFusion/bevfusion/`。BEV 工作分辨率 = `out_size_factor=8` 之于
`grid_size=1440` → **180×180**（lidar config 第 108/110 行）。**与 DepthFusion 的 W×H=180×180 恰好一致。**

```
BEVFusion.extract_feat()                         bevfusion.py:240-284
├─ 图像分支 extract_img_feat()                    bevfusion.py:130-164
│   img_backbone(Swin) → img_neck(GeneralizedLSSFPN) ⇒ F_img [B,N=6,256,32,88]
│   └─ view_transform(...) (autocast float32)     bevfusion.py:153-154
│        = DepthLSSTransform                       depth_lss.py:334-426
│          BaseDepthTransform.forward              depth_lss.py:256-331
│            • raw LiDAR→各相机稀疏深度图 depth[B,N,1,256,704]  depth_lss.py:276-315
│            • get_cam_feats(img,depth): dtransform(1→8→32→64)+depthnet  depth_lss.py:361-380,406-421
│            • bev_pool(geom,x) → collapse Z (cat over Z)  depth_lss.py:116-148,330
│          downsample(/2)                          depth_lss.py:423-426
│        ⇒ 图像 BEV  [B, 80, 180, 180]
├─ LiDAR 分支 extract_pts_feat()                   bevfusion.py:166-173
│   voxelize → BEVFusionSparseEncoder ⇒ LiDAR BEV  [B, 256, 180, 180]
├─ 融合 self.fusion_layer(features)                bevfusion.py:275-276
│   = ConvFuser: cat([img80,lidar256])→Conv→ [B,256,180,180]  transfusion_head.py:28-42
├─ pts_backbone(SECOND) → pts_neck(SECONDFPN)      bevfusion.py:281-282  ⇒ [B,512,180,180]
└─ bbox_head = TransFusionHead                     transfusion_head.py:45+
    num_proposals=200, num_decoder_layers=1, in_channels=512, hidden=128, classes=10
    经 bbox_head.loss()/predict()                   bevfusion.py:286-298 / 203-238
```

**四个插入点**：
- **P1 视角变换内部**（`depth_lss.py` 深度分支）→ 模块 **A=SDG**（及可选 LOG）。
- **P2 融合处**（`bevfusion.py:275-276`）→ 模块 **C=DGF**。
- **P3 检测头之后**（`bevfusion.py` loss/predict，bbox_head 之后）→ 模块 **D=InsFusion**。
- A 的 **LOG** 子项需要 `bev_pool` **之前**的未压扁 3D 体积 `F_c[C,Z,H,W]`（见 §4.2）。

---

## 1. 开关接口与组合

四个 config 布尔开关，**默认全 False**（= 原基线）：

| 开关 | 模块 | 插入点 | 默认 |
| --- | --- | --- | --- |
| `use_sdg` | A: GAFusion SDG | P1 (view transform) | False |
| `use_log` | A: GAFusion LOG（可选/次要） | P1 的 3D 体积处 | False |
| `use_dgf` | C: DepthFusion DGF | P2 (fusion, 替换 ConvFuser) | False |
| `use_insfusion` | D: InsFusion | P3 (head 之后, refine) | False |

**组合顺序固定**：`A(视角变换) → C(融合) → head → D(精修)`。
支持档位：`baseline / +A / +C / +A+C / +A+C+D`。

**开关承载方式（推荐）**：通过「是否构建对应子模块」体现，all-off 时连模块都不实例化、与基线逐字一致：
- `use_sdg/use_log` → 新 `view_transform` 类型（如 `SDGDepthLSSTransform`，继承 `DepthLSSTransform`）；
- `use_dgf` → `fusion_layer` 用新类型 `DGFFuser` 替换 `ConvFuser`；
- `use_insfusion` → `model` 顶层新增可选 `refine_head`（`InsFusionRefineHead`），在 `loss/predict` 条件调用。

**config 接口草案**：
```python
# baseline：用 4xA30 config，不动。
# +A：
model = dict(view_transform=dict(type='SDGDepthLSSTransform', use_sdg=True, use_log=False, ...))
# +C：
model = dict(fusion_layer=dict(type='DGFFuser', use_dgf=True,
                               in_channels=[80, 256], embed_dims=256, num_heads=8, ...))
# +A+C：合并上面两段 override。
# +A+C+D：再加
model = dict(refine_head=dict(type='InsFusionRefineHead', use_insfusion=True,
                              num_query=300, num_layers=2, ...))
```

---

## 2. 依赖链（开 A 之后的耦合）

- **C(DGF)** 的 key/value = 图像 BEV；若 `use_sdg=True`，该图像 BEV 来自 SDG 增强后的 view_transform 输出。
- **D(InsFusion)** 的「raw 相机特征」：**按论文是 2D 图像特征 `F_img`（`img_neck` 输出，透视图），不是 BEV**
  （InsFusion 用 SparseBEV 在 2D 特征上 adaptive sampling）。SDG 作用在 `img_neck` **之后**的视角变换内部，
  **并不改变 `img_neck` 输出**。

  `⚠️ 仍需你定 (D-5)`：你在任务里设「开 A 后 D 的 raw 相机特征取 SDG 增强后特征」——这是一个**tap 点选择**。
  若严格按 InsFusion，D 的相机 raw 特征应是 `img_neck` 的 2D 特征（不受 SDG 影响）；若按你的意图，需把 tap 点
  改到 SDG 增强后的某层（2D 还是 3D 体积 F_c？）。两者会产生不同的 `+A+C+D` 数值，须明确。

---

## 3. 全关数值一致性（铁律 + 断言）

四开关全 False 时，前向与原基线**数值等价**。设计与断言：
1. **不实例化**：all-off 时新子模块为 `None`/未构建；`extract_feat`/`extract_img_feat`/融合/head 后均走基线分支。
2. **运行期断言**：每个 stock 分支入口 `assert not self.use_xxx`；新分支入口 `assert self.use_xxx`。
3. **集成校验测试**（实现后在 4×A30 跑）：把官方融合权重 `…-5239b1af.pth` 载入开关化模型(all-off)，
   val 评测断言 `NDS/mAP` 与基线一致（fp32/amp 容差内）。
4. **参数零影响**：开启后若用「恒等初始化」防早期破坏数值，须注明；但 all-off 路径不依赖它（直接不走新分支）。

---

## 4. 模块 A = GAFusion **SDG**（+LOG 可选）（GAFusion/MGAF §III-B）

> 任务范围：**只做 SDG**；LOG 可选（§4.2）。
> **禁区**（MGAF 中确认存在并排除）：§III-A 额外下采样 + 稀疏高度压缩（式1）、§III-C MSDPT、
> §III-D AFDT(= LGAFT)、§III-E 时序融合、以及 §III-A 的 BiSeNet2 语义分割先验（见下）。

### 4.1 SDG（Sparse Depth Guidance）

**原文式(2)**：
```
R^s_img = P(R_L, R_img)
F_c     = Concat(F_c, R^s_img)          (GAFusion/MGAF Eq.2)
```
P=投影操作；R_L=raw 点云；R_img=多视角图像；R^s_img=稀疏多视角深度图。
流程：raw LiDAR 投影到各相机 → 稀疏多视角深度图 R^s_img → **共享编码器**提深度特征 → 与图像特征 concat
→ 深度感知相机特征 → 视角变换 → **voxel pooling** → 图像 **3D 特征体积 `F_c ∈ ℝ^{C×Z×H×W}`**。

**🔴 关键发现（决定模块 A 在本基线的意义）**：mmdet3d 基线的 `DepthLSSTransform` **已经实现了 SDG 的整条流水线**：
- 投影 raw LiDAR 到各相机得稀疏深度图：`depth_lss.py:276-315`；
- 共享深度编码器 `dtransform`(1→8→32→64)：`depth_lss.py:361-371`；
- 与图像特征 concat 后 `depthnet`：`depth_lss.py:372-380, 406-421`；
- 视角变换 + `bev_pool`。

因此**在本基线上 SDG ≈ 既有行为**。GAFusion 相对本基线的真正增量只有两点：
- (a) **保留 3D 体积 `F_c[C,Z,H,W]`**（stock 在 `bev_pool` 用 `cat over Z` 把 Z collapse 掉了，
  见 `depth_lss.py:146`）——这是 **LOG 的前置条件**；
- (b) 可选 BiSeNet2 语义先验（MGAF 增项）——**默认不做**（属禁区且需额外分割网络/权重）。

`⚠️ 仍需你定 (A-1)`：鉴于上面，`use_sdg` 在本基线**近似 no-op**。建议把模块 A 重新定义为
「**输出/保留 3D 体积 F_c 的深度感知视角变换**」，其主要价值是**服务 `use_log`**；单开 `use_sdg`（不开 LOG）
预计与 baseline 数值接近。请确认此定位；并确认**不**纳入 BiSeNet2 语义先验。

| 项目 | 内容 |
| --- | --- |
| 输入 | F_img `[B,6,256,32,88]`；raw points；投影矩阵 |
| 输出 | （SDG-only）图像 BEV `[B,80,180,180]`（与基线一致）；（为 LOG）保留 `F_c[C,Z,H,W]` |
| 新增可训练参数 | 若沿用 stock dtransform/depthnet：**~0 新增**；若重构共享编码器：少量卷积 |
| 新增 loss | **无**（GAFusion §III-B 的 SDG 未引入独立深度监督 loss） |

### 4.2 LOG（LiDAR Occupancy Guidance，可选）

**原文式(3)**：
```
F_c = Mul(F_c, O_L)                      (GAFusion/MGAF Eq.3)
```
`O_L ∈ ℝ^{1×Z×H×W}`：LiDAR BEV 特征映射到 3D → **占据预测头**估计占据状态得占据体素；
分辨率与 `F_c` **相同**。`Mul`=逐元素乘（带广播，把 1×Z×H×W 广播到 C×Z×H×W）。
输出 `F_c ∈ ℝ^{C×Z×H×W}` = LiDAR 占据引导的图像 3D 特征体积。

**插入难点**：stock 在 `bev_pool`(`depth_lss.py:146`) 已 collapse Z，故 LOG 必须在 **`bev_pool` 之前**拿到
3D 体积 `F_c[C,Z,H,W]`，与 `O_L` 逐元素乘后再 pool。需新增「LiDAR BEV→3D→占据头」支路。

| 项目 | 内容 |
| --- | --- |
| 占据头 | LiDAR BEV→3D→预测 1×Z×H×W 占据；激活/结构 `⚠️ 仍需你定 (A-2)`（论文未给细节） |
| Z 取值 | 须与图像 3D 体积的 Z 对齐；`⚠️ 仍需你定`（取决于 zbound 离散化与 F_c 的 Z） |
| 新增 loss | 论文未明示占据监督 loss；`⚠️ 仍需你定 (A-3)`：是否加占据 GT 监督（需占据标签来源） |

> `use_log` 标为**次要**：因依赖「保留 3D 体积」改造视角变换，实现成本较高；可在 A 之后单独评估。

---

## 5. 模块 C = DepthFusion **DGF**（DepthFusion §III-B）

> 任务范围：**只做 DGF**。**禁区**：不实现 DLF（与模块 D 重叠）。
> **插入点**：新类 `DGFFuser` **替换** `bevfusion.py:275-276` 的 `ConvFuser`（DGF 本身就是全局融合模块）。

### 5.1 深度编码 D（无可训练参数）

**原文式(1)(2)**：
```
p_k = { (x_k, y_k) : d_k },  k ∈ [1, n]                         (Eq.1)
d_k = E( (x_k, y_k), (x_{n/2}, y_{n/2}) ),  k ∈ [1, n]          (Eq.2)
```
E()=欧氏距离；`(x_{n/2}, y_{n/2})` = ego 中心格坐标；n=BEV 格数。深度矩阵 M 存每个格的 d_k
（DLF 用作查表；**DGF 直接用 M 的完整形式**）。深度编码 **`D = 对 M 施 sin/cos`**（ref[28] 正余弦，**无参**）。

### 5.2 全局融合（式3）+ 聚合（式4）

输入：LiDAR BEV `V_GB ∈ ℝ^{W×H×C}`、图像 BEV `I_GB ∈ ℝ^{W×H×C}`；位置编码 `P`（逐元素加到原 BEV）。
```
V̂_GB = softmax( ((V_GB + P) ⊙ D)(I_GB + P)^T / √C ) · I_GB      (Eq.3, 多头交叉注意力)
        ├ query = (V_GB + P) ⊙ D     （深度编码 D 逐元素乘到 LiDAR-BEV 查询，使查询感知深度权重）
        ├ key   = (I_GB + P)
        └ value = I_GB
F_GB  = N( FFN( N(V̂_GB + V_GB) ) + N(V̂_GB + V_GB) )             (Eq.4, 聚合)
        N()=归一化层；FFN()=含两个卷积操作的前馈网络
```

### 5.3 维度与集成

- 论文：`W×H = 180×180`，`C = 128`（**两路 BEV 统一到同一通道 C**），BEVPoolV2 出图像 BEV。
- **我方基线**：图像 BEV=80ch、LiDAR BEV=256ch。集成方案：各自 1×1 投影到统一 `embed_dims`，
  跑 DGF，输出再投影回 **256ch** 以接 `pts_backbone(in_channels=256)`。
- `⚠️ 仍需你定 (C-1)`：统一 `embed_dims` 取 **128**（贴论文）还是 **256**（贴基线、少一次投影）。
- `⚠️ 仍需你定 (C-2)`：`P` 类型——论文只说「positional encoding」，依 ref[28] 推断为**正弦位置编码（无参）**，
  请确认（或改可学习）。

| 项目 | 内容 |
| --- | --- |
| 输入 | LiDAR BEV `[B,256,180,180]` + 图像 BEV `[B,80,180,180]` |
| 输出 | 融合 BEV `[B,256,180,180]` |
| 新增可训练参数 | QKV 投影、FFN(2 卷积)、归一化、通道投影（D 与正弦 P **无参**） |
| 新增 loss | 无（端到端由检测 loss 驱动） |

`⚠️ 显存红旗 (C-3)`：式(3) 是 **180×180=32400 token 的全局交叉注意力**，注意力图 ~ 32400×32400×heads，
fp16 下单是注意力图就可能 **>15GB**，24GB 卡 batch2 **很可能 OACK/OOM**。论文未提及对 BEV 下采样。
须在 4×A30 实测；若 OOM，候选：先降 batch→grad ckpt；或对 BEV token 下采样后再 attn（属优化，需你许可，
因会偏离论文）。**此为 C 的主要风险点。**

---

## 6. 模块 D = InsFusion（InsFusion §3）

> **插入点**：接基线 `TransFusionHead` **之后**（P3），在 `loss/predict` 条件调用 `InsFusionRefineHead`。

### 6.1 抽取三路 query（§3.1）

- **相机路**：`F_img ∈ ℝ^{h×w×C_img}`；定义 **K 个可学习相机 query**（高斯初始化，D_q 维）+
  **SparseBEV[41] adaptive sampling & mixing** → `Q^(0)_img ∈ ℝ^{K×D_q}`。
- **LiDAR 路**：`F_lidar_bev ∈ ℝ^{X×Y×C}`；预测**实例 heatmap** → **top-K 峰值检测（CenterPoint[15]）** →
  编码空间坐标为初始 embedding → `Q^(0)_lidar ∈ ℝ^{K×D_q}`。
- **融合路**：复用既有融合方法的融合实例特征 → `Q^(0)_fusion`（本基线 = **TransFusionHead 的 object query**）。
- **对齐**：模态各一线性层投影到共享隐空间：`Q̂_m = W_m·Q_m + b_m`，`W_m∈ℝ^{D_q×D_q}`，`b_m∈ℝ^{1×D_q}`。

### 6.2 实例精修（§3.2，式1）

```
Q^(0) = [ Q̂_img ; Q̂_lidar ; Q̂_fusion ] ∈ ℝ^{3K×D_q}
Q^(l) = DeformableTransformerLayer( Q^(l-1);
            Flatten(F_img,raw); Flatten(F_lidar_bev,raw); Flatten(F_fusion) )   (Eq.1)
  迭代 L 次；对三个 Key-Value 源**各算一次**可变形注意力，输出**逐元素相加**更新 query。
```
[;;]=拼接。可复用 mmdet 的 Deformable DETR（ref[34]）解码层。

### 6.3 超参与训练（论文 Appendix A）

- **K = 300**（相机/LiDAR 路各 300 初始 query）；**L = 2**（消融 1/2/6 → L=2 最佳：mAP69.33/NDS72.35）。
- **两阶段训练**：① 训 baseline（有官方权重则跳过、直接用）；② 训「extract proposals from img」网络；
  训 InsFusion-增强模型时**加载并冻结 baseline 权重** + 加载 img-proposal 网络权重（**最小微调，低成本**）。
- 原文优化：Adam one-cycle max-lr `2e-5`, wd 0.01, batch 16, 6 epoch, 8×RTX8000。
  （原文 baseline 是 FocalFormer3D / IS-Fusion，非 vanilla BEVFusion，但 §1 声明**兼容所有 BEV 融合模型**。）

### 6.4 假设解析

| 编号 | 论文已解 | 仍需你定（集成） |
| --- | --- | --- |
| D-1 query 复用/另起 | **另起**：相机/LiDAR 各 300 新 query；融合路复用基线 query；三套 concat→3K | 融合路 query = TransFusion 的 200 object query？(`⚠️`) |
| D-2 对齐方式 | 模态线性投影 `Q̂=WQ+b` 后**直接 concat**（无匈牙利匹配） | D_q 取值（论文未给数值）(`⚠️`) |
| D-3 替换/残差、loss | 接 head 后、**冻结 baseline**、残差精修；标准检测 loss 监督精修输出 | `+D` 档训练是否沿用「冻结 baseline+2e-5」还是全端到端(`⚠️ 公平性张力`) |
| D-4 子模块来源 | SparseBEV[41] / CenterPoint[15] | 具体采样点数/层数等超参版本(`⚠️`) |
| D-5 raw 相机特征 | 论文= 2D 图像特征 `F_img`（img_neck 输出） | 你设「SDG 增强后」与论文不一致 → tap 点你定(`⚠️` 见 §2) |

| 项目 | 内容 |
| --- | --- |
| 输入 | F_img(2D) + F_lidar_bev + F_fusion_bev + 三路 proposal |
| 输出 | 精修后的 3D 框/类别（deformable decoder + 预测头） |
| 新增可训练参数 | 600 新 query(2×300)、SparseBEV 采样/混合、LiDAR heatmap 头、模态对齐线性层、L=2 可变形解码层 |
| 新增 loss | 精修输出的标准检测 loss（cls+bbox±heatmap）；`⚠️` 是否额外监督三路 proposal 待定 |

---

## 7. 显存预估（24GB/卡, batch2, amp）

> ⚠️ 不编造 GB 数。下表给**定性增量 + 驱动因素**；精确 GB 须在 4×A30 用
> `torch.cuda.max_memory_allocated()` 实测回填(`PENDING`)。各档**共用同一 batch/累积**以保可比。

| 档位 | 相对增量(定性) | 主要驱动 | 实测峰值 GB |
| --- | --- | --- | --- |
| baseline | 基准 | 主干+LSS+head | `PENDING` |
| +A (SDG) | 极小 | ≈ 既有行为；LOG 时多一条 3D 体积+占据头（中） | `PENDING` |
| +C (DGF) | **大/红旗** | 180×180 全局交叉注意力，注意力图 ~32400² × heads，**OOM 风险最高**（§5.3） | `PENDING` |
| +A+C | 大 | 以 C 为主 | `PENDING` |
| +A+C+D | 大 | 再加 600 新 query + 三源可变形注意力(L=2)；deformable 采样较省显存 | `PENDING` |

**OOM 处理顺序**（保持有效 batch=32、各档可比）：`per-card batch 2→1（accum 4→8）→ activation
checkpointing（img_backbone.with_cp=True）`。C 若仍 OOM，再议是否对 BEV token 下采样（偏离论文，需你许可）。

---

## 8. 代码注释约定（实现阶段）

```python
# GAFusion/MGAF (arXiv:2411.00340 / IEEE TPAMI 2026) Sec.III-B Eq.(2) SDG: 多视角稀疏深度+共享编码器+concat
# GAFusion/MGAF Sec.III-B Eq.(3) LOG: F_c = Mul(F_c, O_L), O_L∈R^{1×Z×H×W}
# DepthFusion (arXiv:2505.07398) Sec.III-B Eq.(3) 全局交叉注意力 / Eq.(4) 聚合
# InsFusion (arXiv:2509.08374) Sec.3.2 Eq.(1): 3K query 可变形精修, 三源 KV 逐元素相加
```

---

## 9. 假设总表（论文已解 vs 仍需你定）

| 编号 | 模块 | 状态 | 内容 |
| --- | --- | --- | --- |
| A-1 | A | ⚠️ 仍需你定 | SDG 在本基线≈no-op；同意把 A 重定义为「保留 3D 体积 F_c 服务 LOG」？ |
| A-bisenet | A | ⚠️ 仍需你定 | 是否纳入 MGAF 的 BiSeNet2 语义先验（默认否） |
| A-2 | A | ⚠️ 仍需你定 | LOG 占据头结构/激活、Z 与 F_c 对齐方式（论文未给） |
| A-3 | A | ⚠️ 仍需你定 | LOG 是否加占据监督 loss 及 GT 来源（论文未明示，倾向无） |
| C-1 | C | ⚠️ 仍需你定 | 统一 embed_dims=128(贴论文) 还是 256(贴基线) |
| C-2 | C | ✅ 论文已解(待确认) | 位置编码 P=正弦(ref28,无参)；深度编码 D=sin/cos(M)(无参) |
| C-3 | C | ⚠️ 显存红旗 | 180×180 全局注意力可能 OOM；是否允许 BEV 下采样(偏离论文) |
| C-fuse | C | ✅ 论文已解 | DGF **替换** ConvFuser（它就是全局融合，式3/4） |
| D-1 | D | ✅+⚠️ | 相机/LiDAR 各 300 新 query；融合路复用基线 query（=TransFusion 200 object query？待确认） |
| D-2 | D | ✅ 论文已解 | 线性投影对齐后 concat（无匈牙利匹配） |
| D-3 | D | ⚠️ 仍需你定 | `+D` 沿用「冻结 baseline+2e-5 两阶段」还是全端到端（与公平性铁律的张力） |
| D-4 | D | ✅ 论文已解 | 相机=SparseBEV[41]，LiDAR=CenterPoint[15] heatmap 峰值 |
| D-5 | D | ⚠️ 仍需你定 | raw 相机特征 tap 点：论文=2D img_neck 特征；你设=SDG 增强后(2D? 3D F_c?) |
| Dq | D | ⚠️ 仍需你定 | query 维度 D_q 数值（论文未给） |

---

## 10. 待你审查/拍板的集成决策清单

1. **A**：接受 SDG 在本基线≈no-op、把模块 A 重定义为「输出/保留 3D 体积 F_c」以服务 LOG？
   是否纳入 BiSeNet2 语义先验（默认否）？LOG 是否加占据监督 loss？
2. **C**：统一 `embed_dims` = 128 还是 256？位置编码用正弦（默认确认）？
   是否预先批准「若 DGF 全局注意力 OOM 则对 BEV token 下采样」这一偏离论文的兜底？
3. **D**：`+D` 档训练沿用论文「冻结 baseline + 2e-5 两阶段」还是改成与 A/C 一致的全端到端（影响公平性对齐）？
   D 的「raw 相机特征」tap 点（2D img_neck vs SDG 增强）？融合路 query 复用 TransFusion 的 200 object query？`D_q` 取值？

> 待你就以上拍板后，下一任务再实现各模块内部（本任务不实现）。
