# Copyright (c) OpenMMLab. All rights reserved.
# [module-A/depth-sup] Visual check of the depth-supervision GT alignment.
#
# Standalone debug tool (does NOT touch training). It runs the real TRAIN
# pipeline (incl. ImageAug3D) on a few nuScenes frames, rebuilds the sparse
# LiDAR depth GT with the EXACT projection used at training time
# (copied verbatim from BaseDepthTransform.forward in depth_lss.py, which
# applies img_aug_matrix / lidar_aug_matrix), and overlays it on the augmented
# image, coloured by depth.
#
# Purpose: eyeball whether the GT points land on object surfaces in the
# *augmented* image and whether the colour (depth) matches near/far objects.
# If points are off-object or depths look wrong, the GT/feature alignment
# (suspicion #1) is broken.
#
# Run from the repo root (needs compiled BEVFusion ops to import the project):
#   python projects/BEVFusion/tools/viz_depth_gt.py \
#       --config projects/BEVFusion/configs/bevfusion_lidar-cam_4xa30_depthsup_nus-3d.py \
#       --num-samples 2 --out-dir outputs/depth_gt_viz
import argparse
import os

import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from mmengine.config import Config  # noqa: E402
from mmengine.registry import init_default_scope  # noqa: E402

# Make `import projects...` work when run as a bare script: a script run only
# puts its own dir on sys.path[0] (not the repo root); dist_train.sh sets
# PYTHONPATH but a direct `python ...` does not. Add the repo root (3 levels up).
import sys  # noqa: E402
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import projects.BEVFusion.bevfusion  # noqa: F401,E402  (register modules)
from mmdet3d.registry import DATASETS  # noqa: E402


def _t(x):
    """to a float64 torch tensor."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().double()
    return torch.as_tensor(np.asarray(x), dtype=torch.float64)


@torch.no_grad()
def project_points(points, lidar2image, img_aug, lidar_aug, image_size):
    """Replicate BaseDepthTransform.forward's LiDAR->image projection for ONE
    sample (depth_lss.py L282-316), verbatim, so the viz reflects the real GT.

    Args:
        points (Tensor): (Npts, >=3) post-pipeline (augmented) LiDAR points.
        lidar2image (Tensor): (Ncam, 4, 4).
        img_aug (Tensor): (Ncam, 4, 4)  image-augmentation matrix.
        lidar_aug (Tensor): (4, 4)      lidar-augmentation matrix.
        image_size (tuple): (iH, iW) of the augmented image.

    Returns:
        coords (Tensor): (Ncam, Npts, 2) pixel [row, col].
        dist   (Tensor): (Ncam, Npts)    camera-frame depth.
        on_img (Tensor): (Ncam, Npts)    bool, point lands inside the image.
    """
    coords = points[:, :3].clone()                                   # (Npts,3)
    # inverse lidar aug
    coords = coords - lidar_aug[:3, 3]
    coords = torch.inverse(lidar_aug[:3, :3]).matmul(coords.transpose(1, 0))
    # lidar2image
    coords = lidar2image[:, :3, :3].matmul(coords)                   # (Ncam,3,Npts)
    coords = coords + lidar2image[:, :3, 3].reshape(-1, 3, 1)
    # depth + perspective divide
    dist = coords[:, 2, :].clone()
    coords[:, 2, :] = torch.clamp(coords[:, 2, :], 1e-5, 1e5)
    coords[:, :2, :] = coords[:, :2, :] / coords[:, 2:3, :]
    # image aug
    coords = img_aug[:, :3, :3].matmul(coords)
    coords = coords + img_aug[:, :3, 3].reshape(-1, 3, 1)
    coords = coords[:, :2, :].transpose(1, 2)                        # (Ncam,Npts,2) [x,y]
    coords = coords[..., [1, 0]]                                     # -> [row,col]
    iH, iW = image_size
    on_img = ((coords[..., 0] < iH) & (coords[..., 0] >= 0)
              & (coords[..., 1] < iW) & (coords[..., 1] >= 0)
              & (dist > 0))
    return coords, dist, on_img


def get_meta(data_sample, key):
    m = data_sample.metainfo
    assert key in m, f'meta key {key!r} not found; have {list(m.keys())}'
    return m[key]


def main():
    p = argparse.ArgumentParser(description='Visualise depth-supervision GT')
    p.add_argument(
        '--config',
        default='projects/BEVFusion/configs/'
        'bevfusion_lidar-cam_4xa30_depthsup_nus-3d.py')
    p.add_argument('--num-samples', type=int, default=2)
    p.add_argument('--out-dir', default='outputs/depth_gt_viz')
    p.add_argument('--point-size', type=float, default=3.0)
    p.add_argument('--dmin', type=float, default=1.0)
    p.add_argument('--dmax', type=float, default=60.0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    init_default_scope('mmdet3d')
    cfg = Config.fromfile(args.config)
    # train_dataloader.dataset is the CBGS wrapper -> includes ImageAug3D
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    print(f'[viz] train dataset length: {len(dataset)}')

    for i in range(args.num_samples):
        sample = dataset[i]
        inputs, data_sample = sample['inputs'], sample['data_samples']
        img = inputs['img']                       # (Ncam,3,H,W), augmented, 0-255
        points = _t(inputs['points'])
        img = img.detach().cpu().float()
        Ncam, _, iH, iW = img.shape

        lidar2image = _t(get_meta(data_sample, 'lidar2img'))      # (Ncam,4,4)
        img_aug = _t(get_meta(data_sample, 'img_aug_matrix'))     # (Ncam,4,4)
        lidar_aug = _t(get_meta(data_sample, 'lidar_aug_matrix')) # (4,4)

        coords, dist, on_img = project_points(
            points, lidar2image, img_aug, lidar_aug, (iH, iW))

        fig, axes = plt.subplots(2, 3, figsize=(22, 9))
        axes = axes.ravel()
        for c in range(min(Ncam, 6)):
            ax = axes[c]
            im = img[c].permute(1, 2, 0).numpy()
            im = np.clip(im, 0, 255).astype(np.uint8)[..., ::-1]  # BGR->RGB
            ax.imshow(im)
            m = on_img[c]
            n_on = int(m.sum())
            if n_on > 0:
                rows = coords[c, m, 0].numpy()
                cols = coords[c, m, 1].numpy()
                dep = dist[c, m].numpy()
                sc = ax.scatter(cols, rows, c=dep, s=args.point_size,
                                cmap='jet', vmin=args.dmin, vmax=args.dmax)
                if c == 0:
                    fig.colorbar(sc, ax=ax, fraction=0.03, label='depth (m)')
            ax.set_title(f'cam {c}  ({n_on} pts on image)')
            ax.axis('off')
        fig.suptitle(
            f'sample {i}: sparse LiDAR depth GT over AUGMENTED image '
            '(points should sit on object surfaces; colour=distance)')
        out = os.path.join(args.out_dir, f'depth_gt_sample{i}.png')
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f'[viz] saved {out}  '
              f'(total on-image points: {int(on_img.sum())})')

    print(f'[viz] done -> {args.out_dir}')


if __name__ == '__main__':
    main()
