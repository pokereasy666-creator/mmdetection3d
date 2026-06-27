# Copyright (c) EP-Fusion M0. All rights reserved.
"""EPFusion —— 冻结 BEVFusion 主干 + PoE 融合 + 教师对照 NLL（M0_PLAN B.2 / C / E）。

继承 BEVFusion（铁律 11：不改上游，新代码全在 projects/EPFusion/）：
- 保留属性名 fusion_layer 给【冻结教师】ConvFuser（load_from 的 fusion_layer.* 免费命中）；
- fusion_mode='poe' → 新建可训练 poe_fuser（PoEFuser）；
  fusion_mode='convfuser'(R0) → 新建可训练 student_fuser（ConvFuser 副本，权重由 R0WeightCopyHook 拷）；
- override train() 持久冻结（铁律 10：mmengine 每 epoch 调 model.train()）；
- override loss() 按 fusion_mode 顶层分叉：poe 走 PoE+L_teach+Λ 日志；convfuser 只走 student_fuser+bbox_loss。
"""
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from mmdet3d.registry import MODELS
from projects.BEVFusion.bevfusion import BEVFusion, ConvFuser

from .poe_fuser import PoEFuser

__all__ = ['EPFusion']


@MODELS.register_module()
class EPFusion(BEVFusion):
    """EP-Fusion M0 检测器。

    Args:
        fusion_mode (str): 'poe'（R1-R3，默认）或 'convfuser'（R0 对照）。
        poe_cfg (dict): PoEFuser 构造参数（poe 模式）。
        w_teach (float): 教师对照 NLL 权重。
        w_fused_mse (float): 可选融合 MSE 锚定旁路（默认 0，关闭）。
        其余 BEVFusion 参数经 **kwargs 透传。
    """

    FROZEN_MODULES = ('img_backbone', 'img_neck', 'view_transform',
                      'pts_voxel_encoder', 'pts_middle_encoder', 'fusion_layer')

    def __init__(self,
                 *args,
                 fusion_mode='poe',
                 poe_cfg=None,
                 w_teach=1.0,
                 w_fused_mse=0.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert fusion_mode in ('poe', 'convfuser')
        self.fusion_mode = fusion_mode
        self.w_teach = float(w_teach)
        self.w_fused_mse = float(w_fused_mse)

        if fusion_mode == 'poe':
            self.poe_fuser = PoEFuser(**(poe_cfg or {}))
        else:  # convfuser (R0)：可训练 ConvFuser 副本，初值由 Hook 从 fusion_layer 拷
            self.student_fuser = ConvFuser(in_channels=[80, 256],
                                           out_channels=256)
        self._freeze_modules()

    # ---------------------------------------------------------------- 冻结
    def _freeze_modules(self):
        for name in self.FROZEN_MODULES:
            m = getattr(self, name, None)
            if m is not None:
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)

    def train(self, mode=True):
        """铁律 10：每次切 train 都强制冻结子模块回 eval（BN 统计不被污染）。"""
        super().train(mode)
        for name in self.FROZEN_MODULES:
            m = getattr(self, name, None)
            if m is not None:
                m.eval()
        return self

    # ---------------------------------------------------------------- 分支 helper
    def _img_feats_2d(self, x):
        """2D 图像特征（backbone+neck+reshape）；对应 bevfusion.py:141-151。"""
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W).contiguous()
        x = self.img_backbone(x)
        x = self.img_neck(x)
        if not isinstance(x, torch.Tensor):
            x = x[0]
        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)
        return x

    def _img_geo(self, metas, ref):
        """构建视角变换所需几何矩阵；对应 bevfusion.py:253-265。ref 提供 dtype/device。"""
        lidar2image, camera_intrinsics, camera2lidar = [], [], []
        img_aug_matrix, lidar_aug_matrix = [], []
        for meta in metas:
            lidar2image.append(meta['lidar2img'])
            camera_intrinsics.append(meta['cam2img'])
            camera2lidar.append(meta['cam2lidar'])
            img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
            lidar_aug_matrix.append(meta.get('lidar_aug_matrix', np.eye(4)))
        return dict(
            lidar2image=ref.new_tensor(np.asarray(lidar2image)),
            camera_intrinsics=ref.new_tensor(np.array(camera_intrinsics)),
            camera2lidar=ref.new_tensor(np.asarray(camera2lidar)),
            img_aug_matrix=ref.new_tensor(np.asarray(img_aug_matrix)),
            lidar_aug_matrix=ref.new_tensor(np.asarray(lidar_aug_matrix)),
            metas=metas)

    def _img_bev(self, feats_2d, points, geo):
        """视角变换 → 相机 BEV [B,80,180,180]；对应 bevfusion.py:153-163（fp32 autocast）。

        相机分支吃点云算稀疏深度，故 points 决定深度输入（corrupt_lidar 下学生/教师不同）。
        """
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            x = self.view_transform(
                feats_2d, deepcopy(points), geo['lidar2image'],
                geo['camera_intrinsics'], geo['camera2lidar'],
                geo['img_aug_matrix'], geo['lidar_aug_matrix'], geo['metas'])
        return x

    def _pts_bev(self, points):
        """LiDAR BEV [B,256,180,180]；调父类 extract_pts_feat（签名核于 bevfusion.py:166-173）。"""
        return self.extract_pts_feat({'points': points})

    # ---------------------------------------------------------------- 前向（predict/clean）
    def extract_feat(self, batch_inputs_dict, batch_input_metas, **kwargs):
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        geo = self._img_geo(batch_input_metas, imgs)
        feats_2d = self._img_feats_2d(imgs)
        F_C = self._img_bev(feats_2d, points, geo)
        F_L = self._pts_bev(points)
        if self.fusion_mode == 'convfuser':
            x = self.student_fuser([F_C, F_L])
        else:
            x = self.poe_fuser(F_C, F_L)['mu_F']
        x = self.pts_backbone(x)
        x = self.pts_neck(x)
        return x

    # ---------------------------------------------------------------- 训练 loss
    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        metas = [item.metainfo for item in batch_data_samples]
        mode = batch_inputs_dict.get('corrupt_mode', 'clean')
        imgs = batch_inputs_dict['imgs']
        points = batch_inputs_dict['points']
        geo = self._img_geo(metas, imgs)

        # 冻结分支前向全程 no_grad（显存红利，P4 证实）；教师 F_T 亦在此算并 detach（铁律 2）
        with torch.no_grad():
            feats_2d = self._img_feats_2d(imgs)
            F_C = self._img_bev(feats_2d, points, geo)
            F_L = self._pts_bev(points)
            F_T = None
            if self.fusion_mode == 'poe':
                if mode == 'corrupt_cam':
                    imgs_clean = batch_inputs_dict['imgs_clean']
                    feats_2d_c = self._img_feats_2d(imgs_clean)
                    F_C_t = self._img_bev(feats_2d_c, points, geo)
                    F_T = self.fusion_layer([F_C_t, F_L]).detach()
                elif mode == 'corrupt_lidar':
                    pts_clean = batch_inputs_dict['points_clean']
                    F_C_t = self._img_bev(feats_2d, pts_clean, geo)
                    F_L_t = self._pts_bev(pts_clean)
                    F_T = self.fusion_layer([F_C_t, F_L_t]).detach()
                else:  # clean
                    F_T = self.fusion_layer([F_C, F_L]).detach()

        # 分支 A：R0（convfuser）—— student_fuser 进计算图、可训练；只算 bbox_loss
        if self.fusion_mode == 'convfuser':
            x = self.student_fuser([F_C, F_L])
            x = self.pts_neck(self.pts_backbone(x))
            return self.bbox_head.loss(x, batch_data_samples)

        # 分支 B：R1-R3（poe）—— PoE 融合 + L_det + L_teach + Λ 日志
        poe_out = self.poe_fuser(F_C, F_L)
        x = self.pts_neck(self.pts_backbone(poe_out['mu_F']))
        losses = self.bbox_head.loss(x, batch_data_samples)

        l_teach, logs = self._teacher_loss(poe_out, F_T, mode)
        losses['loss_teach'] = self.w_teach * l_teach
        losses.update(logs)
        return losses

    # ---------------------------------------------------------------- L_teach + 日志
    def _teacher_loss(self, poe_out, F_T, mode):
        """逐分支高斯 NLL（fp32，丢常数项）；干净样本同样计算（铁律 7）。"""
        F_T = F_T.float()
        pc, pl = poe_out['pc'], poe_out['pl']
        lam_C, lam_L = poe_out['lam_C'], poe_out['lam_L']
        log_C, log_L = poe_out['log_lambda_C'], poe_out['log_lambda_L']
        nll_C = (0.5 * lam_C * (pc - F_T).pow(2)).mean() - 0.5 * log_C.mean()
        nll_L = (0.5 * lam_L * (pl - F_T).pow(2)).mean() - 0.5 * log_L.mean()
        l_teach = nll_C + nll_L
        if self.w_fused_mse > 0:
            l_teach = l_teach + self.w_fused_mse * (poe_out['mu_F'] - F_T).pow(2).mean()
        logs = self._lambda_logs(poe_out, F_T, mode)
        logs['teach_nll_raw'] = l_teach.detach()
        return l_teach, logs

    def _lambda_logs(self, poe_out, F_T, mode):
        """恒定输出 12 个 Λ 键（sum/cnt）+ 4 个坍缩预警键（均仅日志，铁律 8 + all_reduce 死锁规避）。"""
        logs = {}
        lam = {'C': poe_out['lam_C'], 'L': poe_out['lam_L']}
        for m in ('clean', 'corrupt_cam', 'corrupt_lidar'):
            for br in ('C', 'L'):
                v = lam[br]
                if m == mode:
                    logs['lambda_%s_%s_sum' % (br, m)] = v.detach().sum()
                    logs['lambda_%s_%s_cnt' % (br, m)] = v.new_tensor(float(v.numel()))
                else:
                    logs['lambda_%s_%s_sum' % (br, m)] = v.new_tensor(0.0)
                    logs['lambda_%s_%s_cnt' % (br, m)] = v.new_tensor(0.0)
        # 坍缩预警：投影特征逐批方差 + 对 F_T 的逐格余弦均值
        pc, pl = poe_out['pc'].detach(), poe_out['pl'].detach()
        ft = F_T.float()
        logs['proj_var_C'] = pc.var()
        logs['proj_var_L'] = pl.var()
        logs['cos_pc_ft'] = F.cosine_similarity(pc, ft, dim=1).mean()
        logs['cos_pl_ft'] = F.cosine_similarity(pl, ft, dim=1).mean()
        return logs
