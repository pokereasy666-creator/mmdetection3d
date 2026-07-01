# Copyright (c) EP-Fusion M0. All rights reserved.
"""退化损坏注册表（M0_PLAN D.1）。

铁律 15：纯 torch 手写，零新依赖（不 import imagecorruptions / nuScenes-C 等测试族库）。
铁律 5：训练族 TRAIN_CORRUPTIONS 与测试族 TEST_CORRUPTIONS_DOC（仅文档字符串、无实现）分离，
        训练代码物理上无法 import 测试族。
铁律 9：所有损坏的随机性由 make_generator(seed, rank, iter) 派生的 torch.Generator 驱动，
        跨 rank 去相关、整 run 可复现。

约定：
- 图像损坏作用于【归一化前】的 0-255 float 多视角图 [N, C, H, W]（铁律 3）。
- 点云损坏作用于【体素化前】的原始点 [P, >=4]（x,y,z,intensity[,ring]）。
- P5 实测：第 5 维(ring)全 0 不可用 → beam_drop 走俯仰角分箱，不实现 ring 分支。
- 严重度 s ∈ [0, 1]；下列区间均为默认值，由 preprocessor / --cfg-options 覆盖（铁律 17）。
"""
import torch
import torch.nn.functional as F

__all__ = [
    'TRAIN_CORRUPTIONS', 'ZERO_ENTRIES', 'NORMAL_ENTRIES',
    'apply_corruption', 'make_generator',
]


def make_generator(seed, rank, it):
    """由 (seed, rank, iter) 确定性派生 CPU torch.Generator（铁律 9）。"""
    g = torch.Generator()
    g.manual_seed(int(seed) * 1000003 + int(rank) * 9973 + int(it))
    return g


def _randn_like_shape(shape, ref, gen):
    return torch.randn(
        shape, generator=gen, dtype=ref.dtype, device='cpu').to(ref.device)


def _randperm(n, ref, gen):
    return torch.randperm(
        n, generator=gen, device='cpu').to(ref.device)


# ------------------------------------------------------------------ 图像损坏
def gaussian_noise(img, s, gen):
    noise = _randn_like_shape(img.shape, img, gen) * (50.0 * s)
    return torch.clamp(img + noise, 0.0, 255.0)


def brightness_contrast(img, s, gen):
    bright = (torch.rand(1, generator=gen).item() * 2.0 - 1.0) * 64.0 * s
    contrast = 1.0 + (torch.rand(1, generator=gen).item() * 2.0 - 1.0) * 0.6 * s
    mean = img.mean()
    out = (img - mean) * contrast + mean + bright
    return torch.clamp(out, 0.0, 255.0)


def downsample_blur(img, s, gen):
    f = 1.0 + 3.0 * s
    if f <= 1.0:
        return img
    H, W = img.shape[-2], img.shape[-1]
    h2, w2 = max(1, int(H / f)), max(1, int(W / f))
    down = F.interpolate(img, size=(h2, w2), mode='bilinear',
                         align_corners=False)
    up = F.interpolate(down, size=(H, W), mode='bilinear',
                       align_corners=False)
    return up


def occlusion_patches(img, s, gen):
    n_patch = int(round(8 * s))
    if n_patch <= 0:
        return img
    img = img.clone()
    H, W = img.shape[-2], img.shape[-1]
    max_side = max(1, int(0.3 * s * min(H, W)))
    fill = img.mean()
    for _ in range(n_patch):
        ph = int(torch.randint(1, max_side + 1, (1,), generator=gen).item())
        pw = int(torch.randint(1, max_side + 1, (1,), generator=gen).item())
        y0 = int(torch.randint(0, max(1, H - ph + 1), (1,), generator=gen).item())
        x0 = int(torch.randint(0, max(1, W - pw + 1), (1,), generator=gen).item())
        img[..., y0:y0 + ph, x0:x0 + pw] = fill
    return img


def zero_image(img, s, gen):
    """整路置零特殊条目（severity 无关）。"""
    return torch.zeros_like(img)


# ------------------------------------------------------------------ 点云损坏
def random_drop(points, s, gen):
    n = points.shape[0]
    if n == 0:
        return points
    rate = min(max(0.8 * s, 0.0), 0.8)
    n_keep = max(int(round((1.0 - rate) * n)), max(1, int(round(0.2 * n))))
    n_keep = min(n_keep, n)
    perm = _randperm(n, points, gen)[:n_keep]
    return points[perm]


def xyz_jitter(points, s, gen):
    if points.shape[0] == 0:
        return points
    points = points.clone()
    noise = _randn_like_shape(
        (points.shape[0], 3), points, gen) * (0.2 * s)
    points[:, :3] = points[:, :3] + noise
    return points


def beam_drop(points, s, gen):
    """俯仰角分箱回退方案（P5 坐实 ring 不可用）：按 elev 分 32 箱，丢若干整箱。"""
    n_drop = int(round(24 * s))
    if n_drop <= 0 or points.shape[0] == 0:
        return points
    n_bins = 32
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = torch.sqrt(x * x + y * y) + 1e-6
    elev = torch.atan2(z, r)
    emin, emax = elev.min(), elev.max()
    if (emax - emin) < 1e-6:
        return points
    bin_idx = ((elev - emin) / (emax - emin) * (n_bins - 1)).long().clamp(0, n_bins - 1)
    drop_bins = _randperm(n_bins, points, gen)[:min(n_drop, n_bins)]
    drop_mask = torch.zeros(
        points.shape[0], dtype=torch.bool, device=points.device)
    for b in drop_bins.tolist():
        drop_mask |= (bin_idx == b)
    keep = ~drop_mask
    if int(keep.sum()) == 0:
        return points  # 安全：避免清空
    return points[keep]


def intensity_noise(points, s, gen):
    if points.shape[1] < 4 or points.shape[0] == 0:
        return points
    points = points.clone()
    inten = points[:, 3]
    # 鲁棒尺度自适应（分位距优先，退化到 std），不硬编码刻度（P5 未回填 dim3 刻度）
    try:
        q = torch.quantile(inten.float(),
                           torch.tensor([0.25, 0.75], dtype=torch.float32,
                                        device=inten.device))
        scale = float(q[1] - q[0])
    except Exception:
        scale = 0.0
    if scale < 1e-6:
        scale = float(inten.float().std()) + 1e-6
    noise = _randn_like_shape(
        inten.shape, inten, gen) * (0.2 * s * scale)
    points[:, 3] = inten + noise
    return points


def zero_points(points, s, gen):
    """整路置零特殊条目（severity 无关）：保留 0.5% 随机点防 voxelize 空输入崩溃。"""
    n = points.shape[0]
    if n == 0:
        return points
    n_keep = max(1, int(round(0.005 * n)))
    perm = _randperm(n, points, gen)[:n_keep]
    return points[perm]


# ------------------------------------------------------------------ 注册表
TRAIN_CORRUPTIONS = {
    'img': {
        'gaussian_noise': gaussian_noise,
        'brightness_contrast': brightness_contrast,
        'downsample_blur': downsample_blur,
        'occlusion_patches': occlusion_patches,
        'zero_image': zero_image,
    },
    'pts': {
        'random_drop': random_drop,
        'xyz_jitter': xyz_jitter,
        'beam_drop': beam_drop,
        'intensity_noise': intensity_noise,
        'zero_points': zero_points,
    },
}

# 整路置零特殊条目（隶属对应 corrupt_* 模式，severity 无关）
ZERO_ENTRIES = {'img': 'zero_image', 'pts': 'zero_points'}
# 普通损坏（按 severity 参数化，供随机抽样）
NORMAL_ENTRIES = {
    'img': ['gaussian_noise', 'brightness_contrast', 'downsample_blur',
            'occlusion_patches'],
    'pts': ['random_drop', 'xyz_jitter', 'beam_drop', 'intensity_noise'],
}


def apply_corruption(modality, name, tensor, s, gen):
    """单一入口：查表调用 TRAIN_CORRUPTIONS[modality][name]。"""
    return TRAIN_CORRUPTIONS[modality][name](tensor, s, gen)


# ------------------------------------------------------------------ 测试族（仅文档，铁律 5）
TEST_CORRUPTIONS_DOC = """\
TEST_CORRUPTIONS_DOC —— 测试族损坏【仅文档，无实现】（铁律 5）。

以下属测试族（M0/训练阶段禁止实现、禁止 import；M1 nuScenes-C 评测时另行实现并审计）：
  - fog（雾）         : 大气散射模型，深度相关对比度衰减 + airlight。
  - rain（雨）        : 雨纹叠加 + 雨雾联合衰减；点云回波衰减/虚点。
  - snow（雪）        : 雪花粒子遮挡 + 点云强散射噪点。
  - motion_blur（运动模糊）: 方向性卷积核（与自车/目标运动一致）。
  - glare（眩光）     : 强光源过曝 + 光晕。
参考：Dong et al., "Benchmarking Robustness of 3D Object Detection to Common
Corruptions"（nuScenes-C）。进入 M1 评测前须对照其完整清单与本训练族逐项查重
（见 M0_PLAN.md D.1 "M1 前置条款"）。本模块刻意不提供上述任何函数实现。
"""
