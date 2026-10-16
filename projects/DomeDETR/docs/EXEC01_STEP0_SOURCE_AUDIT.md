# 执行文档 #1 · 第 0 步源码核对报告（A1–A9）

- 日期：2026-10-16
- 分支：`claude/dome-detr-query-diag`（基于 `claude/jolly-wozniak-c4YMQ` @ `82c2b4e`）
- 核对对象：本仓库 `projects/BEVFusion`（mmdet3d `1.4.0`，`mmdet3d/version.py`）与 `query_diag.tar.gz` 参考实现
- 执行方式：只读核对。没有部署，没有运行。本会话是云端容器（无 GPU、无 torch、无 nuScenes、无权重），第 1–3 步需要在 4×A30 服务器上执行。
- 原执行文档：`projects/DomeDETR/docs/exec_doc_01_query_init_diagnosis.md`

## 0. 结论

1. **A1–A9 九项全部与参考实现的假设一致。**
2. 核对 A4 调用链时发现参考实现有一个**确定会触发的缺陷（F1）**：`predict_by_feat` → `bbox_coder.decode` 会**原地**把 head 输出里的 `center` 改写成米制坐标。参考实现的 `diag_query_recall.py` 在 `test_step` 返回后才读取 `center`，然后再做一次格子→米制换算，等于换算了两次。结果是 T4 的 `final_ca` / `final_agn` 接近 0，S4 必然失败，R5 无法判读。D2（oracle）不受影响。
3. 还有 3 项流程/环境问题（F2 S5 门禁未自动化、F3 配置/Swin 路径、F7 合并段汇总缺脚本）需要修改参考实现，另有 4 项只需记录。
4. 按文档第 3、11 节，**在你确认前不部署、不运行**。第 4 节是拟改清单，第 5 节是开跑前需要你提供的信息。

## 1. 核对总表

| 编号 | 位置（文件:行号） | 一致？ | 适配方案 |
| --- | --- | --- | --- |
| A1 | `projects/BEVFusion/bevfusion/transfusion_head.py:227-256` | 一致 | 无需适配 |
| A2 | `transfusion_head.py:262` | 一致 | 无需适配 |
| A3 | `transfusion_head.py:155-159, 164-173` | 一致 | 无需适配；运行时由 S2 复核 |
| A4 | `bevfusion.py:233-234`；`transfusion_head.py:338-341, 358-386`；`mmdet3d/models/detectors/base.py:86` | 一致（附 F1） | 假设本身成立；另发现原地改写，需改参考实现（C1） |
| A5 | `transfusion_head.py:129, 227-228` | 一致 | 无需适配 |
| A6 | `configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py:84-91, 125-131` | 一致 | 无需适配 |
| A7 | 融合配置 `:134-180`；`mmdet3d/datasets/det3d_dataset.py:81, 324-325`；`mmdet3d/datasets/transforms/formating.py:246-249`；`mmdet3d/datasets/nuscenes_dataset.py:122-125, 184-187` | 一致 | 无需适配（测试流水线没有 `LoadAnnotations3D`，GT 只能来自 `eval_ann_info`，参考实现优先读它，正确） |
| A8 | `transfusion_head.py:717-731`；`mmdet3d/models/utils/gaussian.py:46, 53` | 一致 | 无需适配 |
| A9 | lidar 配置 `:12-17` | 一致 | 无需适配 |

下面的配置路径都相对于 `projects/BEVFusion/configs/`。"lidar 配置" = `bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py`；"融合配置" = `bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py`；"4xa30 配置" = `bevfusion_lidar-cam_voxel0075_4xa30-amp-accum_nus-3d.py`。两份融合配置都没有覆盖 `bbox_head`，head 参数全部来自 lidar 配置。

## 2. 逐项原文与判断

### A1 query 初始化（`transfusion_head.py`）

```text
229        heatmap = dense_heatmap.detach().sigmoid()
233        local_max_inner = F.max_pool2d(
234            heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
235        local_max[:, :, padding:(-padding),
236                  padding:(-padding)] = local_max_inner
238        if self.test_cfg['dataset'] == 'nuScenes':
239            local_max[:, 8, ] = F.max_pool2d(
240                heatmap[:, 8], kernel_size=1, stride=1, padding=0)
249        heatmap = heatmap * (heatmap == local_max)
253        top_proposals = heatmap.view(batch_size, -1).argsort(
```

- sigmoid：一致。
- `nms_kernel_size`×`nms_kernel_size` 局部极大：一致（配置值 3，`padding=1`）。`local_max` 由 `torch.zeros_like` 初始化（:231），只有内部区被赋值，所以边缘一圈为 0：一致。
- nuScenes 第 8、9 类：`kernel_size=1` 的 max_pool 是恒等映射（第 9 类见 :241-242），等价于不做 NMS：一致。参考实现直接写 `heat[:, 8]`，数值逐位相同。
- 跨类全局 `argsort(descending=True)[..., :num_proposals]`（:253-254）；类别 = `top_proposals // heatmap.shape[-1]`、格子 = `top_proposals % heatmap.shape[-1]`（:255-256），`heatmap.shape[-1]` 在 :250 `view` 之后等于 H·W：一致。
- 参考实现的 `replicate_init` 对同一个 float32 张量、在同一设备上做同样的 argsort，排序结果确定一致，S1 预期 ≈1.0。

### A2 `query_labels`（`transfusion_head.py`）

```text
262        self.query_labels = top_proposals_class
```

一致。只在 `forward_single` 赋值，`predict_by_feat`（:365）读取。

### A3 `create_2D_grid` 与 `bev_pos`（`transfusion_head.py`）

```text
155        x_size = self.test_cfg['grid_size'][0] // self.test_cfg[
156            'out_size_factor']
159        self.bev_pos = self.create_2D_grid(x_size, y_size)
165        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
167        batch_x, batch_y = torch.meshgrid(
168            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
169        batch_x = batch_x + 0.5
170        batch_y = batch_y + 0.5
171        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
172        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
```

- `torch.meshgrid` 没有传 `indexing`，torch 2.0.1（服务器实测版本，见 `projects/EPFusion/docs/M0_RECON_REPORT.md` §1）默认是 `'ij'`，只会打印 UserWarning。所以 `batch_x[i, j] = i`，展平后索引 `i·W + j` 对应 `(i+0.5, j+0.5)`：一致。
- "行 r 对应 x 轴"有三处佐证：
  - 稀疏编码器 dense 输出是 `[N, C, H, W, D]`，其中 `sparse_shape=[1440,1440,41]` 是 (x, y, z)，permute 后 BEV 的 dim2 是 x（`sparse_encoder.py:142-146`）。
  - 解码 `center[:, 0]` 换算成 x（`utils.py:77-80`）。
  - 训练目标的行号就是 x（见 A8）。
- 结论一致。运行时由 S2（`bev_pos transposed=False`）复核。

### A4 预测调用链与最终分数

```text
bevfusion.py:234             outputs = self.bbox_head.predict(feats, batch_input_metas)
transfusion_head.py:339          preds_dicts = self(batch_feats, batch_input_metas)
transfusion_head.py:340          res = self.predict_by_feat(preds_dicts, batch_input_metas)
transfusion_head.py:360              batch_score = preds_dict[0]['heatmap'][
transfusion_head.py:361                  ..., -self.num_proposals:].sigmoid()
transfusion_head.py:364              one_hot = F.one_hot(
transfusion_head.py:365                  self.query_labels,
transfusion_head.py:367              batch_score = batch_score * preds_dict[0][
transfusion_head.py:368                  'query_heatmap_score'] * one_hot
transfusion_head.py:370              batch_center = preds_dict[0]['center'][..., -self.num_proposals:]
```

- head 通过 `self(...)` 调用，经过 `nn.Module.__call__` 进入 `self.forward`，所以实例属性替换的 `forward` 会生效。`Base3DDetector.forward` 在 predict 模式下调用 `self.predict(...)`（`base.py:86`），所以对 `BEVFusion.predict` 的实例属性替换也会生效：一致。
- 最终分数 = sigmoid(预测 heatmap) × `query_heatmap_score` × one_hot(`query_labels`)，类别在初始化时锁定：一致。
- **附带发现 F1（见第 3 节）**：:370 的切片是视图，随后 `bbox_coder.decode` 原地改写它。

### A5 `heatmap_head` 调用（`transfusion_head.py`）

```text
129        self.heatmap_head = nn.Sequential(*layers)
227        with torch.autocast('cuda', enabled=False):
228            dense_heatmap = self.heatmap_head(fusion_feat.float())
```

- `projects/BEVFusion` 内只有 :228 一处调用，每次 forward 调用一次，并且是以模块形式调用，所以 forward hook 会触发：一致。
- autocast 关闭且输入 `.float()`，输出是 float32 logits：一致。
- 形状：`num_classes=10`（lidar 配置 :88）；BEV 边长 = 1440 / 8 = 180，即 `[B, 10, 180, 180]`：一致。

### A6 `test_cfg` 与 `num_proposals`（lidar 配置）

```text
84         num_proposals=200,
89         nms_kernel_size=3,
91         num_decoder_layers=1,
125        test_cfg=dict(
126            dataset='nuScenes',
127            grid_size=[1440, 1440, 41],
128            out_size_factor=8,
129            voxel_size=[0.075, 0.075],
130            pc_range=[-54.0, -54.0],
131            nms_type=None),
```

- 全部一致。`bbox_coder` 的 `pc_range` / `out_size_factor` / `voxel_size` 与 `test_cfg` 相同（:134-141）。
- `num_decoder_layers=1` 且 `auxiliary=True`，所以 `center[..., -200:]` 就是唯一一层解码器的输出。
- `nms_type=None`：后处理不做 circle NMS。

### A7 测试流水线与 `eval_ann_info`

```text
融合配置:154-161      dict(type='ImageAug3D', ... rot_lim=[0.0, 0.0], rand_flip=False, is_train=False),   (节选，合并换行)
融合配置:162-164      dict(type='PointsRangeFilter', point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),   (合并换行)
det3d_dataset.py:81                  load_eval_anns: bool = True,
det3d_dataset.py:324         if self.test_mode and self.load_eval_anns:
det3d_dataset.py:325             info['eval_ann_info'] = self.parse_ann_info(info)
formating.py:246         if 'eval_ann_info' in results:
formating.py:247             data_sample.eval_ann_info = results['eval_ann_info']
nuscenes_dataset.py:184                  gt_bboxes_3d = LiDARInstance3DBoxes(
nuscenes_dataset.py:187                      origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
```

- 测试流水线（融合配置 :134-173）依次是：`BEVLoadMultiViewImageFromFiles` → `LoadPointsFromFile` → `LoadPointsFromMultiSweeps` → `ImageAug3D`（`is_train=False`，只处理图像）→ `PointsRangeFilter` → `Pack3DDetInputs`。**点云上没有旋转、缩放、平移、翻转**：一致。4xa30 配置只重写了 `train_pipeline`，测试流水线原样继承。
- val 数据集：`test_mode=True`、`box_type_3d='LiDAR'`（lidar 配置 :303-304）；没有覆盖 `load_eval_anns`，取默认值 True。`eval_ann_info` 在 `parse_data_info` 中、在流水线之前生成，不经过任何变换，由 `Pack3DDetInputs` 原样挂到 data_sample 上。
- 框是 `LiDARInstance3DBoxes`（LiDAR 坐标系，origin 只影响 z）。`gt_labels_3d` 是 int64 ndarray，非目标类为 −1：nuScenes 没有调用 `_remove_dontcare`，参考实现 `diag_common.py:189` 用 `labels >= 0` 过滤，处理正确。
- val 没有设 `use_valid_flag`，按 `num_lidar_pts > 0` 过滤（`nuscenes_dataset.py:122-125`）。
- 注意：测试流水线**没有** `LoadAnnotations3D`，所以 `gt_instances_3d` 为空。参考实现先读 `eval_ann_info`，正好走对路径；它的回退分支（`gt_instances_3d`）在本配置下不会被用到。

### A8 训练目标的高斯中心（`transfusion_head.py`）

```text
717                coor_x = ((x - pc_range[0]) / voxel_size[0] /
718                          self.train_cfg['out_size_factor'])
719                coor_y = ((y - pc_range[1]) / voxel_size[1] /
720                          self.train_cfg['out_size_factor'])
725                center_int = center.to(torch.int32)
727                # original
728                # draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]], center_int, radius) # noqa: E501
729                # NOTE: fix
730                draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]],
731                                      center_int[[1, 0]], radius)
```

- `draw_heatmap_gaussian` 把 `center[0]` 当列、`center[1]` 当行（`gaussian.py:46, 53`）。传入 `[coor_y, coor_x]` 后，行号 = x、列号 = y：与修正写法一致。
- 取整方式：训练目标用 `.to(int32)`（向零截断），参考实现 `xy_to_flat` 用 `floor`。对网格内的 GT（坐标 ≥ 0）两者相同；网格外的 GT 被 `valid` 掩码剔除。等价。

### A9 `class_names`（lidar 配置）

```text
12 class_names = [
13     'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
14     'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
15 ]
17 metainfo = dict(classes=class_names)
```

索引 8 = pedestrian，9 = traffic_cone：一致。`predict_by_feat` 中 nuScenes 的 task 定义（`transfusion_head.py:396-407`）也是 `pedestrian→[8]`、`traffic_cone→[9]`。`NUS_CLASS_RANGE` 与 `detection_cvpr_2019` 一致（`nuscenes_metric.py:97` 默认 eval_version）。

## 3. 额外发现

### F1（缺陷，阻断 D1 的 T4 / S4 / R5）：`decode` 原地改写 `center`

```text
transfusion_head.py:370            batch_center = preds_dict[0]['center'][..., -self.num_proposals:]
utils.py:77        center[:,
utils.py:78               0, :] = center[:,
utils.py:79                              0, :] * self.out_size_factor * self.voxel_size[
utils.py:80                                  0] + self.pc_range[0]
```

- `[..., -200:]` 是基本切片，得到的是视图。`decode` 对它做 `__setitem__`，会写回 head 输出 dict 里的 `center`（`dim` 也会被原地 `exp`，见 :85-87）。源码作者在 `get_targets_single` 里特意用 `copy.deepcopy` 规避这个问题（:586-588 注释 "don't change the network outputs"），说明这是已知行为。
- 参考实现的执行顺序：替换后的 `forward` 把 `out` 存进 `STATE['head_out']` → `predict_by_feat` 原地改写 → `test_step` 返回 → `diag_query_recall.py:222` 读取 `out['center'][..., -P:]`，此时已经是米制坐标，再调用 `grid_to_xy` 又换算一次：`x' = 0.6·x − 54`。对 |x| ≤ 54 m 的目标，偏差都在 32 m 以上，T4 的 final 两列 ≈ 0，**S4 必然失败**。
- 不受影响的部分：
  - D1 的初始 query 部分：位置来自 `bev_pos`，类别来自 `query_labels`，都没有被改写。
  - D2：`head_out_modifier` 在 `forward` 内部、`predict_by_feat` 之前执行，`query_heatmap_score` 的还原能正确传到最终分数。
- mock 测试没发现它，是因为 mock 只按 `forward_single` 编写，没有包含 `predict_by_feat`。
- 修复见 C1。这属于"解码器 center 单位换算有误"（S4 的失败说明），修复的是坐标 bug，没有改动任何统计口径。

### F2（流程）：`run_diag.sh oracle` 一次连跑 5 组，S5 门禁没有落实

文档第 8 节要求 base 组先跑，S5 不通过就不跑后四组，但脚本把 5 组串成一条命令。修复见 C2。

### F3（环境）：配置默认值与 Swin 初始化路径

- `run_diag.sh` 的默认 `CONFIG` 是官方融合配置，其中 Swin 的 `init_cfg` 是 GitHub URL。`BEVFusion.__init__` 在构造时就调用 `img_backbone.init_weights()`（`bevfusion.py:63, 115-117`），所以离线服务器上构建模型时会尝试下载，直接失败（RECON 附录"坑 1"）。
- 4xa30 配置把这个路径钉成了服务器绝对路径 `/212022085500129/mmdetection3d-claude-jolly-wozniak-c4YMQ/checkpoints/swint-nuimages-pretrained.pth`（:41）。只要在原服务器目录下运行就没有问题。
- 参考实现不支持传 `--cfg-options`，路径失效时没有办法绕过。修复见 C3。

### F4（记录，不改）：2 卡 DDP 时帧数统计多 1

`DefaultSampler(round_up=True)` 会把 6019 帧补到 6020，oracle 的 `frames` / `injected_cells` 会多算 1 帧。官方评测在汇总时会截断，mAP/NDS 不受影响。

### F5（记录，不改）：S5 的已知值精度不够

RECON 记录的基线是 NDS 0.696 / mAP 0.665（3 位小数），S5 的容差是 0.1 pp = 0.001，舍入误差就可能吃掉一半容差。需要你提供 `baseline1` epoch_19 验证日志里的 4 位小数值。

### F6（记录，不改）：S3 判据与脚本内置警告不是同一个量

- 文档 S3：0–20 m、class-aware、@2 m、≥ 0.5。
- 脚本内置 `[WARNING]`：近距 class-agnostic、@4 m、< 0.5。
- 两者都要看。C4 的汇总脚本会把 S3 的数值单独打印出来。

### F7（记录 + 新增脚本）：合并段汇总

第 2 步要求的 "<30 m / ≥30 m" 四个数需要从 npz 另算，参考实现里没有这个脚本。见 C4（新增脚本，不改主脚本）。

### F8（记录，不改）：checkpoint 用 `strict=False` 加载

mmengine 以 `strict=False` 加载，spconv 布局不匹配时只会打 warning。`baseline1/epoch_19.pth` 是在本环境里训练出来的，布局一致。如果改用官方权重，需要先过 `tools/convert_spconv_weights.py`。开跑后请检查日志里没有 `size mismatch` / `missing keys`。

## 4. 拟改清单（待确认，未实施）

所有改动都只在 `tools/query_diag/` 内，不碰 `projects/BEVFusion`。

| 编号 | 文件 | 改动 | 原因 |
| --- | --- | --- | --- |
| C1 | `diag_common.py`、`diag_query_recall.py` | 在替换后的 `forward` 里、`predict_by_feat` 之前，把 `center` 存一份快照 `STATE['center_grid']`（`detach().clone()`）；T4 改读这份快照 | F1 |
| C2 | `run_diag.sh` + 新增 `diag_gate_s5.py` | `oracle` 先跑 base，然后按 `BASE_MAP` / `BASE_NDS` 环境变量（已知值）自动判 S5（容差 0.001）；不通过或没设置时就停，不跑后四组。文档里的命令不变 | F2 |
| C3 | `run_diag.sh` | `CONFIG` 改为必填（去掉默认的官方配置）；新增可选的 `CFG_OPTS`，透传给 `--cfg-options`（默认空，不改变 baseline） | F3 |
| C4 | 新增 `diag_merge_bins.py`；`run_diag.sh` 增加 `merge` 入口 | 从 `recall_per_gt.npz` 计算 "<30 m" 和 "≥30 m" 段的四个数，同时打印 S1、S3、S4、R6 用到的数值，输出 `recall_merged.json` / `.txt` | F7、F6 |
| C5 | `run_diag.sh` | 每次启动前用 `nvidia-smi -i $CUDA_VISIBLE_DEVICES` 检查目标卡上有没有计算进程，有就退出 | 文档 §3 "不得抢占"；CLAUDE.md 铁律 13 |
| C6 | `diag_query_recall.py`、`diag_oracle_query.py` | 在 json 里追加 `runtime` 字段：总耗时、`torch.cuda.max_memory_allocated/reserved`、torch/mmcv/mmengine/mmdet3d 版本、仓库 `VERSION` 第一行（不调用 git） | 文档 §6.3、§10 要求回传耗时、显存和环境信息 |

实施后的门禁：所有 .py 做非 ASCII 检查，并用 `ast.parse(..., feature_version=(3, 8))` 解析（服务器是 Python 3.8）；`run_diag.sh` 过 `bash -n`。参考实现原样已经通过了这三项（预检结果：5 个文件非 ASCII 行数都是 0，ast 和 `bash -n` 都通过）。

## 5. 开跑前需要你确认或提供

1. **对本报告和 C1–C6 的确认**（C5 可选）。
2. `<CONFIG>`：建议 `projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_4xa30-amp-accum_nus-3d.py`。
3. `<CKPT>`：建议 `work_dirs/baseline1/epoch_19.pth`（RECON §1 锁定的复现权重），请给出服务器上的绝对路径。
4. `<mAP>` / `<NDS>`：RECON 记录的是 0.665 / 0.696，请从 epoch_19 验证日志补 4 位小数（F5）。
5. 部署方式：建议只把 `tools/query_diag/` 拷进服务器上现有的 `mmdetection3d-claude-jolly-wozniak-c4YMQ` 目录运行。那里的 `data/nuscenes`、`work_dirs/baseline1` 和 Swin 绝对路径都已经可用，可以避开 F3。
6. R1 阈值（1.0 或 0.8 pp）：第 3 步开跑前确认即可。
