# modify from https://github.com/mit-han-lab/bevfusion
from typing import Tuple

import torch
from torch import nn

from mmdet3d.registry import MODELS
from .depth_sup import depth_bce_loss, drop_depth_input
from .ops import bev_pool


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor(
        [row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2]
                           for row in [xbound, ybound, zbound]])
    return dx, bx, nx


class BaseViewTransform(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.feature_size = feature_size
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound

        dx, bx, nx = gen_dx_bx(self.xbound, self.ybound, self.zbound)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.C = out_channels
        self.frustum = self.create_frustum()
        self.D = self.frustum.shape[0]
        self.fp16_enabled = False

    def create_frustum(self):
        iH, iW = self.image_size
        fH, fW = self.feature_size

        ds = (
            torch.arange(*self.dbound,
                         dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW))
        D, _, _ = ds.shape

        xs = (
            torch.linspace(0, iW - 1, fW,
                           dtype=torch.float).view(1, 1, fW).expand(D, fH, fW))
        ys = (
            torch.linspace(0, iH - 1, fH,
                           dtype=torch.float).view(1, fH, 1).expand(D, fH, fW))

        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(
        self,
        camera2lidar_rots,
        camera2lidar_trans,
        intrins,
        post_rots,
        post_trans,
        **kwargs,
    ):
        B, N, _ = camera2lidar_trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = (
            torch.inverse(post_rots).view(B, N, 1, 1, 1, 3,
                                          3).matmul(points.unsqueeze(-1)))
        # cam_to_lidar
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        combine = camera2lidar_rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)

        if 'extra_rots' in kwargs:
            extra_rots = kwargs['extra_rots']
            points = (
                extra_rots.view(B, 1, 1, 1, 1, 3,
                                3).repeat(1, N, 1, 1, 1, 1, 1).matmul(
                                    points.unsqueeze(-1)).squeeze(-1))
        if 'extra_trans' in kwargs:
            extra_trans = kwargs['extra_trans']
            points += extra_trans.view(B, 1, 1, 1, 1,
                                       3).repeat(1, N, 1, 1, 1, 1)

        return points

    def get_cam_feats(self, x):
        raise NotImplementedError

    def bev_pool(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) /
                      self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([
            torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
            for ix in range(B)
        ])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = ((geom_feats[:, 0] >= 0)
                & (geom_feats[:, 0] < self.nx[0])
                & (geom_feats[:, 1] >= 0)
                & (geom_feats[:, 1] < self.nx[1])
                & (geom_feats[:, 2] >= 0)
                & (geom_feats[:, 2] < self.nx[2]))
        x = x[kept]
        geom_feats = geom_feats[kept]

        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)

        return final

    def forward(
        self,
        img,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        **kwargs,
    ):
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        x = self.get_cam_feats(img)
        x = self.bev_pool(geom, x)
        return x


@MODELS.register_module()
class LSSTransform(BaseViewTransform):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        self.depthnet = nn.Conv2d(in_channels, self.D + self.C, 1)
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    def get_cam_feats(self, x):
        B, N, C, fH, fW = x.shape

        x = x.view(B * N, C, fH, fW)

        x = self.depthnet(x)
        depth = x[:, :self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x

    def forward(self, *args, **kwargs):
        x = super().forward(*args, **kwargs)
        x = self.downsample(x)
        return x


class BaseDepthTransform(BaseViewTransform):

    def forward(
        self,
        img,
        points,
        lidar2image,
        cam_intrinsic,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        **kwargs,
    ):
        intrins = cam_intrinsic[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        batch_size = len(points)
        depth = torch.zeros(batch_size, img.shape[1], 1,
                            *self.image_size).to(points[0].device)

        for b in range(batch_size):
            cur_coords = points[b][:, :3]
            cur_img_aug_matrix = img_aug_matrix[b]
            cur_lidar_aug_matrix = lidar_aug_matrix[b]
            cur_lidar2image = lidar2image[b]

            # inverse aug
            cur_coords -= cur_lidar_aug_matrix[:3, 3]
            cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                cur_coords.transpose(1, 0))
            # lidar2image
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
            # get 2d coords
            dist = cur_coords[:, 2, :]
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

            # imgaug
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)

            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]]

            on_img = ((cur_coords[..., 0] < self.image_size[0])
                      & (cur_coords[..., 0] >= 0)
                      & (cur_coords[..., 1] < self.image_size[1])
                      & (cur_coords[..., 1] >= 0))
            for c in range(on_img.shape[0]):
                masked_coords = cur_coords[c, on_img[c]].long()
                masked_dist = dist[c, on_img[c]]
                depth = depth.to(masked_dist.dtype)
                depth[b, c, 0, masked_coords[:, 0],
                      masked_coords[:, 1]] = masked_dist

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]
        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        x = self.get_cam_feats(img, depth)
        x = self.bev_pool(geom, x)
        return x


@MODELS.register_module()
class DepthLSSTransform(BaseDepthTransform):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
        use_depth_sup: bool = False,
        depth_loss_weight: float = 3.0,
        depth_input_keep_ratio: float = 1.0,
    ) -> None:
        """Compared with `LSSTransform`, `DepthLSSTransform` adds sparse depth
        information from lidar points into the inputs of the `depthnet`.

        [module-A/depth-sup] ``use_depth_sup`` (default False) toggles BEVDepth
        (arXiv:2206.10092) explicit depth supervision. When False, this module
        is byte-identical to the baseline: nothing is cached, no loss is added,
        and no parameters are introduced.

        [module-A/depth-sup-v2] ``depth_input_keep_ratio`` (default 1.0 = OFF,
        i.e. unchanged v1 behavior) closes the input-reconstruction shortcut.
        The stock ``DepthLSSTransform`` feeds the *same* sparse LiDAR depth both
        as the ``depthnet`` INPUT and (under v1 supervision) as the GT TARGET, so
        the supervised pixels already have their answer in the input -> the net
        can learn to copy the input instead of inferring depth (leakage). With
        ``keep_ratio < 1.0``, TRAINING randomly drops ``1 - keep_ratio`` of the
        input LiDAR points fed to ``dtransform`` while the supervision GT keeps
        the FULL projection (see ``get_cam_feats``). Most supervised pixels then
        have NO answer in the input, forcing depth *completion*, not copying.
        Inference is UNCHANGED (dropout is train-only), and no parameters are
        added. Only meaningful together with ``use_depth_sup=True``.
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        self.dtransform = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.D + self.C, 1),
        )
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

        # [module-A/depth-sup] BEVDepth (arXiv:2206.10092) explicit depth
        # supervision. Adds NO trainable parameters: it only supervises the
        # depth logits the `depthnet` already predicts. Caches below are
        # populated ONLY when use_depth_sup is True (see get_cam_feats); they
        # are plain attributes -> never enter parameters()/state_dict.
        self.use_depth_sup = use_depth_sup
        self.depth_loss_weight = depth_loss_weight
        # [module-A/depth-sup-v2] input-dropout keep ratio (1.0 == OFF).
        assert 0.0 < depth_input_keep_ratio <= 1.0, (
            f'depth_input_keep_ratio must be in (0, 1], got '
            f'{depth_input_keep_ratio}')
        self.depth_input_keep_ratio = depth_input_keep_ratio
        self._depth_pred_logits = None  # (B*N, D, fH, fW), pre-softmax
        self._depth_gt = None           # (B*N, 1, iH, iW), sparse LiDAR depth

    def get_cam_feats(self, x, d):
        B, N, C, fH, fW = x.shape

        d = d.view(B * N, *d.shape[2:])
        x = x.view(B * N, C, fH, fW)

        if self.use_depth_sup:
            # [module-A/depth-sup] stash the sparse LiDAR depth GT at full
            # image resolution BEFORE `dtransform` overwrites `d`. This is the
            # FULL projection -- the supervision target is never dropped.
            self._depth_gt = d
            if self.training:
                # [module-A/depth-sup-v2] drop (1 - keep_ratio) of the input
                # LiDAR points fed to `dtransform`, so most SUPERVISED pixels
                # have no answer in the input -> the depthnet must COMPLETE
                # depth, not copy it (closes the leakage shortcut). `d` is
                # reassigned to a dropped copy; `self._depth_gt` still refers to
                # the full tensor above (drop_depth_input never mutates its
                # input), so the GT is unaffected. keep_ratio >= 1.0 is a no-op.
                d = drop_depth_input(d, self.depth_input_keep_ratio)

        d = self.dtransform(d)
        x = torch.cat([d, x], dim=1)
        x = self.depthnet(x)

        if self.use_depth_sup:
            # [module-A/depth-sup] stash the PRE-softmax depth logits for BCE
            # (the depth below is numerically identical to the baseline; this
            # only keeps an extra reference to the logits).
            self._depth_pred_logits = x[:, :self.D]
            depth = self._depth_pred_logits.softmax(dim=1)
        else:
            depth = x[:, :self.D].softmax(dim=1)  # baseline, byte-identical
        x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x

    def get_depth_loss(self):
        """[module-A/depth-sup] BEVDepth (arXiv:2206.10092) depth supervision.

        Softmax (over the depth-bin dim, matching the forward lift) + BCE
        between the stashed pre-softmax depth logits and the discretized sparse
        LiDAR depth GT, with official BEVDepth normalization (per-valid-pixel
        sum over bins, averaged over valid pixels). Only pixels with a LiDAR
        point in ``[d_min, d_max)`` are supervised (the sparse mask).
        Reads AND clears the caches; returns the weighted scalar loss.
        """
        logits, gt = self._depth_pred_logits, self._depth_gt
        assert logits is not None and gt is not None, (
            'get_depth_loss called without cached depth tensors; ensure '
            'use_depth_sup=True and a forward ran before loss().')
        self._depth_pred_logits = None
        self._depth_gt = None
        return depth_bce_loss(logits, gt, self.image_size, self.feature_size,
                              self.dbound, self.D, self.depth_loss_weight)

    def forward(self, *args, **kwargs):
        x = super().forward(*args, **kwargs)
        x = self.downsample(x)
        return x
