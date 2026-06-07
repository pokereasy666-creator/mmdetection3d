# DESIGN — 三模块开关化设计（GAFusion-SDG / DepthFusion-DGF / InsFusion）

> 范围：**仅设计**，不实现任何模块内部（本任务约束）。
> 文中标 `🔲 待补` 的公式/内部维度需要**论文原文段落**才能定稿
> （用户将粘贴 GAFusion §3.2 / DepthFusion §III-B / InsFusion §3）。
> 标 **`⚠️ ASSUMPTION NEEDED`** 的是论文/任务未写清、需你确认的设计点。
> 实现阶段：每个重构组件的代码注释须引用对应论文**节号/公式号**（约定见 §8）。

---

## 0. 基线前向回顾（插入点的坐标系）

文件均在 `projects/BEVFusion/bevfusion/` 下。BEV 工作分辨率 = `out_size_factor=8` 之于
`grid_size=1440` → **180×180**（见 lidar config 第 108/110 行）。

```
BEVFusion.extract_feat()                         bevfusion.py:240-284
├─ 图像分支 extract_img_feat()                    bevfusion.py:130-164
│   img_backbone(Swin) → img_neck(GeneralizedLSSFPN) → [B,N=6,256,32,88]
│   └─ view_transform(...)  (autocast float32)    bevfusion.py:153-154
│        = DepthLSSTransform                       depth_lss.py:334-426
│          BaseDepthTransform.forward              depth_lss.py:256-331
│            • raw LiDAR → 各相机稀疏深度图 depth[B,N,1,256,704]  depth_lss.py:276-315
│            • get_cam_feats(img, depth)           depth_lss.py:406-421
│                dtransform(depth):1→8→32→64       depth_lss.py:361-371
│                depthnet(cat[64,img256]→D+C)      depth_lss.py:372-380,414
│            • bev_pool(geom, x) → collapse Z      depth_lss.py:116-148,330
│          downsample(/2)                          depth_lss.py:423-426
│        ⇒ 图像 BEV  [B, 80, 180, 180]
├─ LiDAR 分支 extract_pts_feat()                   bevfusion.py:166-173
│   voxelize → BEVFusionSparseEncoder → 稠密 BEV
│        ⇒ LiDAR BEV [B, 256, 180, 180]
├─ 融合 self.fusion_layer(features)                bevfusion.py:275-276
│   = ConvFuser: cat([img80, lidar256])→Conv→ [B,256,180,180]  transfusion_head.py:28-42
├─ pts_backbone(SECOND) → pts_neck(SECONDFPN)      bevfusion.py:281-282  ⇒ [B,512,180,180]
└─ bbox_head = TransFusionHead                     transfusion_head.py:45+
    num_proposals=200, num_decoder_layers=1, in_channels=512, hidden=128, classes=10
    经 bbox_head.loss()/predict() 出框                bevfusion.py:286-298 / 203-238
```

**四个插入点**（与三模块对应）：
- **P1 视角变换内部**（`depth_lss.py` 深度分支）→ 模块 **A=SDG**（及可选 LOG）。
- **P2 融合处**（`bevfusion.py:275-276` 的 `fusion_layer`）→ 模块 **C=DGF**。
- **P3 检测头之后**（`bevfusion.py` loss/predict，`bbox_head` 之后）→ 模块 **D=InsFusion**。
- （A 的 LOG 子项需要 bev_pool **之前**的未压扁 3D 体积，见 §4.2。）

---

## 1. 开关接口与组合

四个 config 布尔开关，**默认全 False**（= 原基线）：

| 开关 | 模块 | 插入点 | 默认 |
| --- | --- | --- | --- |
| `use_sdg` | A: GAFusion SDG | P1 (view transform) | False |
| `use_log` | A: GAFusion LOG（可选/次要） | P1 之后的 3D 体积 | False |
| `use_dgf` | C: DepthFusion DGF | P2 (fusion) | False |
| `use_insfusion` | D: InsFusion | P3 (head 之后) | False |

**组合顺序固定**：`A(视角变换) → C(融合) → head → D(精修)`。
支持的实验档：`baseline / +A / +C / +A+C / +A+C+D`。

**开关承载方式（推荐）**：开关不放裸 flag，而是通过「是否构建对应子模块」体现，
保证 all-off 时连模块都不实例化、代码路径与基线逐字一致：
- `use_sdg/use_log` → 作为 `view_transform` 的新增子模块/参数（新 view_transform 类型或在
  `DepthLSSTransform` 上加可选分支）；
- `use_dgf` → `fusion_layer` 用新类型（如 `DGFFuser`）替换 `ConvFuser`；
- `use_insfusion` → `model` 顶层新增可选 `refine_head`，在 `BEVFusion.loss/predict` 中条件调用。

**config 示例（接口草案）**：
```python
# baseline：用 §EXPERIMENTS 的 4xA30 config，不动。
# +A：
model = dict(
    view_transform=dict(type='SDGDepthLSSTransform', use_sdg=True, use_log=False, ...))
# +C：
model = dict(
    fusion_layer=dict(type='DGFFuser', use_dgf=True,
                      in_channels=[80, 256], out_channels=256, ...))
# +A+C：同时给出上面两段 override。
# +A+C+D：再加
model = dict(
    refine_head=dict(type='InsFusionRefineHead', use_insfusion=True, ...))
```
> 最终接口字段在实现时定稿；本表先固定**语义与默认值**。

---

## 2. 依赖链（开 A 之后的关键耦合）

**开 A 后，下游"图像侧"输入都变成 SDG 增强后的特征**：
- **C(DGF)** 的 key/value = 图像 BEV，若 `use_sdg=True` 则该图像 BEV 来自 SDG 增强后的 view_transform 输出；
- **D(InsFusion)** 的「raw 相机特征」同理来自 SDG 增强后的特征。

  `⚠️ ASSUMPTION NEEDED (D-5)`：D 所谓「raw 相机特征」到底指 `img_neck` 输出([B,N,256,32,88])
  还是 `view_transform` 输出的图像 BEV([B,80,180,180])？开 A 后取哪一层的 SDG 增强结果？

因此评测必须按档位区分：`+C` 与 `+A+C` 的 C 输入不同，不可混用。

---

## 3. 全关数值一致性（铁律 + 断言）

要求：四开关全 False 时，前向与原基线**数值等价**。

设计与断言：
1. **不实例化**：all-off 时新子模块为 `None`/未构建；`extract_feat`、`extract_img_feat`、
   融合、head 之后的代码走与基线**完全相同**的分支。
2. **运行期断言**：在每个 stock 分支入口加
   `assert not self.use_xxx, '<flag> on but stock path taken'`，反向也然。
3. **集成校验测试**（实现后在 4×A30 跑）：把官方融合权重 `…-5239b1af.pth` 载入开关化模型（all-off），
   在 val 上评测，断言 `NDS/mAP` 与基线**逐位一致**（允许 fp32/amp 容差）。
4. **参数零影响**：任何"恒等初始化"技巧（如残差分支末层零初始化）若用于保证开启后初期不破坏数值，
   须在文档与注释注明——但 all-off 路径不应依赖它（应直接不走新分支）。

---

## 4. 模块 A = GAFusion **SDG**（arXiv 2411.00340 §3.2）

> 任务范围：**只做 SDG**；LOG 可选（§4.2）。
> **禁区**：不实现 MSDPT、额外下采样/稀疏高度压缩、LGAFT、时序融合。

### 4.1 SDG（Sparse Depth Guidance）

**任务简述的流水线**：raw LiDAR 投影到各相机 → 稀疏多视角深度图 → 共享深度编码器 →
与图像特征 concat → 深度感知相机特征 → 送 LSS 视角变换 → voxel pooling。

**插入点**：`depth_lss.py` 的深度分支（`BaseDepthTransform.forward` L256-331 与
`DepthLSSTransform.get_cam_feats` L406-421）。

**⚠️ 关键 ASSUMPTION NEEDED (A-SDG-1)** — **SDG 与现有实现的关系**：
现有 `DepthLSSTransform` **本身就已经**：把 raw LiDAR 投影到各相机得稀疏深度图
(L276-315)、过一个共享深度编码器 `dtransform`(1→8→32→64, L361-371)、与图像特征 concat
后过 `depthnet` (L372-380) 得到深度感知相机特征、再 LSS + bev_pool。
**这与 SDG 的描述高度重合**。需确认 SDG 相对 stock 的**实质增量**是什么：
- (a) 仅是同一思想，复现≈保持现状（则 `use_sdg` 近乎 no-op，需重新界定）？
- (b) 不同的深度编码器结构 / 多视角间共享方式 / 深度概率监督？
- (c) 引入**显式深度监督 loss**（用投影深度做 GT）？

| 项目 | 内容 |
| --- | --- |
| 输入 | 图像特征 `[B,N=6,256,32,88]`；raw LiDAR points（list）；投影矩阵（lidar2image 等） |
| 输出 | 深度感知图像 BEV `[B,80,180,180]`（维度与基线一致，便于 all-off 对齐） |
| 新增可训练参数 | `🔲 待补`：SDG 深度编码器/融合层的结构与通道（需 §3.2 原文） |
| 新增 loss | `🔲 待补 / ⚠️ A-SDG-1(c)`：是否有深度监督 loss 及其形式 |
| 公式 | `🔲 待补`：§3.2 中 SDG 的式子（深度编码、concat、深度感知特征生成） |

### 4.2 LOG（LiDAR Occupancy Guidance，可选）

**简述**：LiDAR BEV → 3D → 占据预测头 → 占据体素 `O_L ∈ [1,Z,H,W]` →
与图像 3D 特征体积逐元素乘（论文**式(2)**）。

**插入难点 / ⚠️ ASSUMPTION NEEDED (A-LOG-1)**：stock 流程在 `bev_pool` 内**已把 Z 维 collapse**
(`final = cat(x.unbind(dim=2),1)`，`depth_lss.py:146`)。LOG 的逐元素乘需要**未压扁的 3D 体积**
`O_I ∈ [·,Z,H,W]`，故插入点必须在 `bev_pool` 之前（拿到 `[B,N,D,fH,fW,C]` 或重排的 3D BEV 体积），
或单独维护一条 3D 体积支路。需确认论文里 `O_I` 的确切定义与坐标系。

| 项目 | 内容 |
| --- | --- |
| 公式(2) | `🔲 待补`：`O_I' = O_I ⊙ O_L` 的精确形式与广播规则 |
| 占据头 | `🔲 待补`：结构、输出激活、Z/H/W 取值 |
| 新增 loss | `🔲 待补 / ⚠️`：是否有占据监督 loss（需占据 GT 来源） |

> LOG 标为**次要**：`use_log` 接口先占位；若 §4.2 插入难点过大，实现时可仅保留 SDG。

---

## 5. 模块 C = DepthFusion **DGF**（arXiv 2505.07398 §III-B）

> 任务范围：**只做 DGF**。**禁区**：不实现 DLF（与模块 D 重叠，避免消融冗余）。

**插入点**：替换/包裹 `bevfusion.py:275-276` 的 `fusion_layer`（新类 `DGFFuser`）。
输入两路 BEV：LiDAR `[B,256,180,180]`、图像 `[B,80,180,180]`；输出融合 BEV，
**通道须仍为 256** 以接 `pts_backbone`(in_channels=256)。

**深度编码 D（无可训练参数）**：深度矩阵 = 各 BEV 格到 ego 中心格的欧氏距离，施加正余弦。
```
🔲 待补（需 §III-B 原文）：
  距离矩阵 M[i,j] = ||p_{ij} - p_ego||₂ 的精确定义；
  正余弦编码 D = [sin(M/τ_k), cos(M/τ_k)]_k 的频率/维度/归一化尺度。
```

**全局融合（论文式3）**（按任务简述）：
```
query     = (LiDAR_BEV + pos_enc) · D
key,value = image_BEV + pos_enc
out       = MultiHeadCrossAttention(query, key, value)
🔲 待补：式(3) 的精确张量形式、序列化(HW→token)方式、head 数、d_model。
```

**聚合（论文式4）**（按任务简述）：`F = N(FFN(N(V̂+V)) + N(V̂+V))`，N=归一化。
```
🔲 待补：V̂ / V 的精确定义（V̂=注意力输出？V=image_BEV 投影？）、N 是 LayerNorm 还是 BN。
```

**⚠️ ASSUMPTION NEEDED**：
- (C-1) query(256ch) 与 key/value(80ch) 通道/序列对齐：是否各自线性投影到统一 `d_model`？
- (C-2) `DGFFuser` 是**替换** ConvFuser，还是 DGF 之后再接一个 1×1/Conv 回到 256 通道（或保留 ConvFuser 串接）？
- (C-3) 位置编码 `pos_enc` 类型（可学习 vs 正弦）与是否含可训练参数。
- (C-4) 深度矩阵 `D` 与 query 的结合是逐元素乘、缩放，还是作为 attention bias？「·」的语义需核实。

| 项目 | 内容 |
| --- | --- |
| 输入 | LiDAR BEV `[B,256,180,180]` + 图像 BEV `[B,80,180,180]` |
| 输出 | 融合 BEV `[B,256,180,180]` |
| 新增可训练参数 | 交叉注意力 QKV 投影、FFN、归一化（D 与正余弦 pos_enc **无参**） |
| 新增 loss | 预计无（融合模块，端到端由检测 loss 驱动）；`🔲 待核实` |

---

## 6. 模块 D = InsFusion（arXiv 2509.08374 §3 全部）

**插入点**：接在基线 `TransFusionHead` **之后**（P3）；在 `BEVFusion.loss/predict`
(`bevfusion.py:286-298 / 203-238`) 中条件调用一个 `refine_head`。

**三路 proposal**（按任务简述）：
1. **raw 相机**：K 个可学习 query + **SparseBEV 的 adaptive sampling & adaptive mixing**（开源，注明来源）。
2. **raw LiDAR BEV**：实例 heatmap + top-K 峰值检测（**CenterPoint 式**，注明来源；
   仓库已有 `mmdet3d.models.dense_heads.centerpoint_head`、`circle_nms`、`draw_heatmap_gaussian`，
   见 `transfusion_head.py:16-17` 的 import，可复用）。
3. **融合 BEV**：来自基线主干/头的 proposal。

对齐后用注意力在 **raw 特征**上精修。

**依赖来源（须在代码注释与文档注明）**：
- 相机路：SparseBEV adaptive sampling+mixing（开源实现）。
- LiDAR 路：CenterPoint heatmap 峰值检测。

**⚠️ ASSUMPTION NEEDED（核心，用户已点名）**：
- (D-1) **query 复用 vs 另起**：D 的 query 与 `TransFusionHead` 现有 object query
  （`num_proposals=200`）是**复用同一套**，还是 D **另起 K 个新 query**？— 直接影响参数与匹配逻辑。
- (D-2) 三路 proposal 的数量(K)与**对齐方式**（匈牙利匹配 / 按 BEV 位置 / NMS 去重？）。
- (D-3) D 是**替换** head 输出还是**残差精修**；梯度如何回流；
  新增 loss 是**复用** TransFusion 的 cls/bbox/heatmap，还是新设监督？
- (D-4) SparseBEV 采样点数/层数、CenterPoint heatmap 超参的具体取值与来源版本。
- (D-5) 见 §2：「raw 相机特征」指哪一层，开 A 后取 SDG 增强结果的哪一层。

| 项目 | 内容 |
| --- | --- |
| 输入 | 基线 head 输出的 proposal + raw 相机特征 + raw LiDAR BEV + 融合 BEV |
| 输出 | 精修后的 3D 框/类别（替换或细化 head 预测） |
| 新增可训练参数 | K 个 query（若另起）、adaptive sampling/mixing、LiDAR heatmap 头、精修注意力 |
| 新增 loss | `🔲 待补 / ⚠️ D-3`：精修分支的监督（cls+bbox±heatmap） |
| 公式 | `🔲 待补`：§3 的采样/混合/对齐/精修式子 |

---

## 7. 显存预估（24GB/卡, batch2, amp）

> ⚠️ 不编造 GB 数。下表给**定性增量与驱动因素**；精确 GB 须在 4×A30 用
> `torch.cuda.max_memory_allocated()` 实测回填（`PENDING`）。所有档位**共用同一 batch/累积**以保可比。

| 档位 | 相对 baseline 显存增量(定性) | 主要驱动 | 实测峰值 GB |
| --- | --- | --- | --- |
| baseline | 基准 | 主干+LSS+head | `PENDING` |
| +A (SDG) | 小 | 深度编码器/融合卷积，体量小 | `PENDING` |
| +C (DGF) | 中 | BEV 180×180=32400 token 的交叉注意力（注意力图 ~ N_q×N_kv） | `PENDING` |
| +A+C | 中 | A(小)+C(中) | `PENDING` |
| +A+C+D | 中→大 | 再加 K query + adaptive sampling/mixing + 精修注意力 | `PENDING` |

**OOM 处理顺序**（保持有效 batch=32 与各档可比）：
`降 per-card batch（2→1，accum 4→8）→ 开 activation checkpointing（如 img_backbone.with_cp=True）`。
C 的注意力可考虑对 BEV 下采样后再做（但属优化，非本任务）。

---

## 8. 代码注释约定（实现阶段）

每个重构组件的 docstring/关键行注释须引用论文坐标，例如：
```python
# GAFusion (arXiv:2411.00340) Sec.3.2 SDG: 共享深度编码器 + 深度感知相机特征
# DepthFusion (arXiv:2505.07398) Sec.III-B Eq.(3) 全局交叉注意力 / Eq.(4) 聚合
# InsFusion (arXiv:2509.08374) Sec.3: 三路 proposal 抽取 + raw 特征精修
```

---

## 9. ASSUMPTION NEEDED 汇总（待你确认）

| 编号 | 模块 | 待确认 |
| --- | --- | --- |
| A-SDG-1 | A | SDG 相对 stock `DepthLSSTransform`(已含稀疏深度+dtransform+depthnet) 的实质增量；是否带深度监督 loss |
| A-SDG-2 | A | SDG 深度编码器结构/通道、concat 维度 |
| A-LOG-1 | A | LOG 需未压扁 3D 体积 `O_I`，而 stock 在 bev_pool 已 collapse Z；插入点/支路如何留 3D |
| A-LOG-2 | A | 占据头结构、O_L 的 Z/H/W、是否带占据监督 loss 与 GT 来源 |
| C-1 | C | query(256ch)/key,value(80ch) 通道与序列对齐方式 |
| C-2 | C | DGF 替换 ConvFuser 还是串接；输出回 256 通道方式 |
| C-3 | C | 位置编码类型及是否含可训练参数 |
| C-4 | C | 深度矩阵 D 与 query 的「·」语义（乘/缩放/attn-bias）、归一化尺度与正余弦频率 |
| D-1 | D | D 的 query 复用 head object query 还是另起 K 个 |
| D-2 | D | 三路 proposal 数量 K 与对齐方式 |
| D-3 | D | D 替换 vs 残差精修；新增 loss 复用还是新设 |
| D-4 | D | SparseBEV / CenterPoint 子模块超参与来源版本 |
| D-5 | D | 「raw 相机特征」指哪一层；开 A 后取 SDG 增强结果哪一层 |

---

## 10. 待你提供（解锁公式定稿）

为把上文 `🔲 待补` 处写成可引用式号的精确公式/维度，请粘贴：
1. **GAFusion** §3.2「LiDAR Guidance」中 SDG（与可选 LOG、式(2)）的原文段落；
2. **DepthFusion** §III-B「Depth-GFusion (DGF)」的深度编码、式(3)、式(4) 原文；
3. **InsFusion** §3 全节（三路 proposal、对齐、精修，及任何 loss/维度定义）。

收到后我会回填公式块并逐条消解 §9 中可由原文确定的 ASSUMPTION，再做最终提交。
