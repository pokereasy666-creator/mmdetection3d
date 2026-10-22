# -*- coding: utf-8 -*-
"""QueryLocTransFusionHead (exec doc #3 v1.1).

A subclass of projects/BEVFusion TransFusionHead (which stays unchanged):

* query_init  'cell'   positions = cell centres (TransFusion);
              'dark'   positions = DARK decode of the 3x3 neighbourhood
                       (subcell.py, shared with step 0a; method M-a);
              'offset' positions = cell centre + learned offset delta of the
                       selected cell (method M-b); delta is detached before
                       use, the offset head is trained only by a dense L1.
* feat_sampling 'bilinear' query features sampled at the continuous
                       position (exact gather on a cell centre); 'cell' keeps
                       the peak-cell gather (used by the step 0b probe).
* unbiased_heatmap     M-a target: Gaussian around the continuous GT centre,
                       GT cell = 1 (focal-loss positives unchanged).
* get_targets          per-sample slicing of the layer-concatenated
                       predictions (audit T1: the original indexes
                       preds_dict[i] and fails for num_decoder_layers > 1);
                       identical to the original for one layer.
* copy init            when a checkpoint lacks decoder / prediction-head
                       layer i >= 1, it is filled from layer 0 (D3).

Top-K selection, query_labels and query_heatmap_score are the original
code (copied verbatim below); only positions and features change.
"""
import numpy as np
import torch
import torch.nn.functional as F
from mmdet.models.utils import multi_apply
from torch import nn

from mmdet3d.models import gaussian_radius
from mmdet3d.registry import MODELS
from projects.BEVFusion.bevfusion.transfusion_head import TransFusionHead
from .subcell import bilinear_gather, dark_offsets, neighborhood3x3

_NO_CAT = ('dense_heatmap', 'dense_heatmap_old', 'query_heatmap_score',
           'dense_offset')


@MODELS.register_module()
class QueryLocTransFusionHead(TransFusionHead):

    def __init__(self,
                 *args,
                 query_init='cell',
                 feat_sampling='bilinear',
                 dark_clamp=0.5,
                 unbiased_heatmap=False,
                 offset_channels=128,
                 offset_range=3.0,
                 offset_sup_radius=3,
                 loss_offset_weight=0.25,
                 copy_init_layers=True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert query_init in ('cell', 'dark', 'offset'), query_init
        assert feat_sampling in ('bilinear', 'cell'), feat_sampling
        self.query_init = query_init
        self.feat_sampling = feat_sampling
        self.dark_clamp = float(dark_clamp)
        self.unbiased_heatmap = bool(unbiased_heatmap)
        self.offset_range = float(offset_range)
        self.offset_sup_radius = int(offset_sup_radius)
        self.loss_offset_weight = float(loss_offset_weight)
        self.query_pos_init = None          # [B, P, 2], read by diagnostics
        self.query_cells_init = None        # [B, P] selected flat cells
        self.debug_check_gather = False     # S-M0 bitwise check
        self.debug_gather_equal = None
        self.copied_layers = []
        if query_init == 'offset':
            hc, oc = self.class_encoding.out_channels, int(offset_channels)
            self.offset_head = nn.Sequential(
                nn.Conv2d(hc, oc, 3, padding=1, bias=False),
                nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
                nn.Conv2d(oc, oc, 3, padding=1, bias=False),
                nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
                nn.Conv2d(oc, 2, 1))
            nn.init.zeros_(self.offset_head[-1].weight)
            nn.init.zeros_(self.offset_head[-1].bias)
            self.init_bn_momentum()
        if copy_init_layers and self.num_decoder_layers > 1:
            self._register_load_state_dict_pre_hook(self._copy_init_hook)

    # ------------------------------------------------------------------
    # D3: fill missing decoder layers from layer 0 at load time
    # ------------------------------------------------------------------
    def _copy_init_hook(self, state_dict, prefix, local_metadata, strict,
                        missing_keys, unexpected_keys, error_msgs):
        for sub in ('decoder', 'prediction_heads'):
            src = '%s%s.0.' % (prefix, sub)
            src_items = [(k, v) for k, v in state_dict.items()
                         if k.startswith(src)]
            if not src_items:
                continue
            for i in range(1, self.num_decoder_layers):
                dst = '%s%s.%d.' % (prefix, sub, i)
                if any(k.startswith(dst) for k in state_dict):
                    continue
                for k, v in src_items:
                    state_dict[dst + k[len(src):]] = v.clone()
                self.copied_layers.append(dst)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward_single(self, inputs, metas):
        batch_size = inputs.shape[0]
        fusion_feat = self.shared_conv(inputs)

        #################################
        # image to BEV
        #################################
        fusion_feat_flatten = fusion_feat.view(batch_size,
                                               fusion_feat.shape[1],
                                               -1)  # [BS, C, H*W]
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(fusion_feat.device)

        #################################
        # query initialization
        #################################
        with torch.autocast('cuda', enabled=False):
            dense_heatmap = self.heatmap_head(fusion_feat.float())
        heatmap = dense_heatmap.detach().sigmoid()
        heat_raw = heatmap                  # QueryLoc: pre-NMS sigmoid
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        # equals to nms radius = voxel_size * out_size_factor * kenel_size
        local_max_inner = F.max_pool2d(
            heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
        local_max[:, :, padding:(-padding),
                  padding:(-padding)] = local_max_inner
        # for Pedestrian & Traffic_cone in nuScenes
        if self.test_cfg['dataset'] == 'nuScenes':
            local_max[:, 8, ] = F.max_pool2d(
                heatmap[:, 8], kernel_size=1, stride=1, padding=0)
            local_max[:, 9, ] = F.max_pool2d(
                heatmap[:, 9], kernel_size=1, stride=1, padding=0)
        elif self.test_cfg[
                'dataset'] == 'Waymo':  # for Pedestrian & Cyclist in Waymo
            local_max[:, 1, ] = F.max_pool2d(
                heatmap[:, 1], kernel_size=1, stride=1, padding=0)
            local_max[:, 2, ] = F.max_pool2d(
                heatmap[:, 2], kernel_size=1, stride=1, padding=0)
        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)

        # top num_proposals among all classes
        top_proposals = heatmap.view(batch_size, -1).argsort(
            dim=-1, descending=True)[..., :self.num_proposals]
        top_proposals_class = top_proposals // heatmap.shape[-1]
        top_proposals_index = top_proposals % heatmap.shape[-1]

        # QueryLoc: initial positions (grid units) and query features
        cell_pos = bev_pos.gather(
            index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(
                -1, -1, bev_pos.shape[-1]),
            dim=1,
        )
        H, W = fusion_feat.shape[-2:]
        dense_offset = None
        if self.query_init == 'cell':
            query_pos = cell_pos
        elif self.query_init == 'dark':
            nb, _, border = neighborhood3x3(heat_raw, top_proposals_index,
                                            top_proposals_class)
            off, _, _ = dark_offsets(nb, border, clamp=self.dark_clamp)
            query_pos = cell_pos + off.to(cell_pos.dtype)   # row = x
        else:
            with torch.autocast('cuda', enabled=False):
                dense_offset = torch.tanh(self.offset_head(
                    fusion_feat.float())) * self.offset_range  # [B, 2, H, W]
            delta = dense_offset.view(batch_size, 2, -1).gather(
                2, top_proposals_index[:, None, :].expand(-1, 2, -1))
            query_pos = cell_pos + delta.permute(0, 2, 1).detach()
        cell_feat = None
        if self.query_init == 'cell' or self.feat_sampling == 'cell' or \
                self.debug_check_gather:
            cell_feat = fusion_feat_flatten.gather(
                index=top_proposals_index[:, None, :].expand(
                    -1, fusion_feat_flatten.shape[1], -1),
                dim=-1,
            )
        if self.query_init == 'cell' or self.feat_sampling == 'cell':
            query_feat = cell_feat
        else:
            query_feat = bilinear_gather(fusion_feat_flatten, query_pos, H, W)
        if self.debug_check_gather:
            self.debug_gather_equal = (
                bool(torch.equal(query_feat, cell_feat)),
                bool(torch.equal(query_pos, cell_pos)))
        self.query_pos_init = query_pos.detach()
        self.query_cells_init = top_proposals_index.detach()
        self.query_labels = top_proposals_class

        # add category embedding
        one_hot = F.one_hot(
            top_proposals_class,
            num_classes=self.num_classes).permute(0, 2, 1)
        query_cat_encoding = self.class_encoding(one_hot.float())
        query_feat += query_cat_encoding

        #################################
        # transformer decoder layer (Fusion feature as K,V)
        #################################
        ret_dicts = []
        for i in range(self.num_decoder_layers):
            # Transformer Decoder Layer
            # :param query: B C Pq    :param query_pos: B Pq 3/6
            query_feat = self.decoder[i](
                query_feat,
                key=fusion_feat_flatten,
                query_pos=query_pos,
                key_pos=bev_pos)

            # Prediction
            res_layer = self.prediction_heads[i](query_feat)
            res_layer['center'] = res_layer['center'] + query_pos.permute(
                0, 2, 1)
            ret_dicts.append(res_layer)

            # for next level positional embedding
            query_pos = res_layer['center'].detach().clone().permute(0, 2, 1)

        ret_dicts[0]['query_heatmap_score'] = heatmap.gather(
            index=top_proposals_index[:,
                                      None, :].expand(-1, self.num_classes,
                                                      -1),
            dim=-1,
        )  # [bs, num_classes, num_proposals]
        ret_dicts[0]['dense_heatmap'] = dense_heatmap
        if dense_offset is not None:
            ret_dicts[0]['dense_offset'] = dense_offset

        if self.auxiliary is False:
            # only return the results of last decoder layer
            return [ret_dicts[-1]]

        # return all the layer's results for auxiliary superivison
        new_res = {}
        for key in ret_dicts[0].keys():
            if key not in _NO_CAT:
                new_res[key] = torch.cat(
                    [ret_dict[key] for ret_dict in ret_dicts], dim=-1)
            else:
                new_res[key] = ret_dicts[0][key]
        return [new_res]

    # ------------------------------------------------------------------
    # targets and losses
    # ------------------------------------------------------------------
    def get_targets(self, batch_gt_instances_3d, preds_dict):
        """As TransFusionHead.get_targets, but slices every sample out of
        the (already layer-concatenated) predictions instead of indexing
        preds_dict[i] per layer."""
        list_of_pred_dict = []
        for batch_idx in range(len(batch_gt_instances_3d)):
            list_of_pred_dict.append({
                key: value[batch_idx:batch_idx + 1]
                for key, value in preds_dict[0].items()
            })
        res_tuple = multi_apply(
            self.get_targets_single,
            batch_gt_instances_3d,
            list_of_pred_dict,
            np.arange(len(batch_gt_instances_3d)),
        )
        labels = torch.cat(res_tuple[0], dim=0)
        label_weights = torch.cat(res_tuple[1], dim=0)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        ious = torch.cat(res_tuple[4], dim=0)
        num_pos = np.sum(res_tuple[5])
        matched_ious = np.mean(res_tuple[6])
        heatmap = torch.cat(res_tuple[7], dim=0)
        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            ious,
            num_pos,
            matched_ious,
            heatmap,
        )

    def get_targets_single(self, gt_instances_3d, preds_dict, batch_idx):
        res = super().get_targets_single(gt_instances_3d, preds_dict,
                                         batch_idx)
        if not self.unbiased_heatmap:
            return res
        heatmap = self.unbiased_heatmap_target(gt_instances_3d,
                                               res[7].device)
        return tuple(res[:7]) + (heatmap[None], )

    def _grid(self):
        """Grid constants exactly as get_targets_single builds them."""
        tc = self.train_cfg
        return (torch.tensor(tc['grid_size']),
                torch.tensor(tc['point_cloud_range']),
                torch.tensor(tc['voxel_size']), tc['out_size_factor'])

    def unbiased_heatmap_target(self, gt_instances_3d, device):
        """M-a: same radius / sigma / window / max-merge as
        get_targets_single + draw_heatmap_gaussian, but the Gaussian is
        centred on the continuous GT position; the GT cell is set to 1.
        The grid arithmetic is copied from get_targets_single, so a GT on
        a cell centre gives the original target (up to the float32 rounding
        of its grid coordinate; positives identical)."""
        gt_bboxes_3d = gt_instances_3d.bboxes_3d
        gt_labels_3d = gt_instances_3d.labels_3d
        gt_bboxes_3d = torch.cat(
            [gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]],
            dim=1).to(device)
        grid_size, pc_range, voxel_size, osf = self._grid()
        feature_map_size = (grid_size[:2] // osf)  # [x_len, y_len]
        heatmap = gt_bboxes_3d.new_zeros(self.num_classes,
                                         feature_map_size[1],
                                         feature_map_size[0])
        height, width = heatmap.shape[1:]
        eps = np.finfo(np.float64).eps
        for idx in range(len(gt_bboxes_3d)):
            width_ = gt_bboxes_3d[idx][3]
            length = gt_bboxes_3d[idx][4]
            width_ = width_ / voxel_size[0] / osf
            length = length / voxel_size[1] / osf
            if not (width_ > 0 and length > 0):
                continue
            radius = gaussian_radius(
                (length, width_),
                min_overlap=self.train_cfg['gaussian_overlap'])
            radius = max(self.train_cfg['min_radius'], int(radius))
            x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]
            coor_x = ((x - pc_range[0]) / voxel_size[0] / osf)
            coor_y = ((y - pc_range[1]) / voxel_size[1] / osf)
            center = torch.tensor([coor_x, coor_y], dtype=torch.float32)
            center_int = center.to(torch.int32)
            # rows <-> x, cols <-> y (draw_heatmap_gaussian(center_int[[1,0]]))
            row, col = int(center_int[0]), int(center_int[1])
            top, bottom = min(row, radius), min(height - row, radius + 1)
            left, right = min(col, radius), min(width - col, radius + 1)
            if top + bottom <= 0 or left + right <= 0:
                continue
            rr = torch.arange(row - top, row + bottom, dtype=torch.float64)
            cc = torch.arange(col - left, col + right, dtype=torch.float64)
            cx, cy = float(center[0]), float(center[1])
            sigma = (2 * radius + 1) / 6.0
            d2 = (rr[:, None] + 0.5 - cx)**2 + (cc[None, :] + 0.5 - cy)**2
            g = torch.exp(-d2 / (2 * sigma * sigma))
            g[(rr == row)[:, None] & (cc == col)[None, :]] = 1.0
            g[g < eps * g.max()] = 0
            sl = heatmap[gt_labels_3d[idx], row - top:row + bottom,
                         col - left:col + right]
            torch.max(sl, g.to(device, torch.float32), out=sl)
        return heatmap

    def offset_l1(self, dense_offset, batch_gt_instances_3d):
        """M-b dense L1: cells within Chebyshev radius R of every GT cell;
        target = continuous GT grid coordinate - cell centre, clipped to
        +-offset_range; a cell claimed by several GT goes to the nearest GT
        centre; per cell |dx| + |dy|, summed over the batch and divided by
        the number of supervised cells."""
        B, _, H, W = dense_offset.shape
        dev = dense_offset.device
        _, pc_range, voxel_size, osf = self._grid()
        pc_range, voxel_size = pc_range.to(dev), voxel_size.to(dev)
        R, clip = self.offset_sup_radius, self.offset_range
        d = torch.arange(-R, R + 1, device=dev)
        total = dense_offset.new_zeros(())
        count = 0
        for b, gt in enumerate(batch_gt_instances_3d):
            if len(gt.bboxes_3d) == 0:
                continue
            ctr = gt.bboxes_3d.gravity_center[:, :2].to(dev).float()
            gx = (ctr[:, 0] - pc_range[0]) / voxel_size[0] / osf
            gy = (ctr[:, 1] - pc_range[1]) / voxel_size[1] / osf
            rr = gx.floor().long()[:, None, None] + d[None, :, None]
            cc = gy.floor().long()[:, None, None] + d[None, None, :]
            rr, cc = torch.broadcast_tensors(rr, cc)
            dx = gx[:, None, None] - (rr.float() + 0.5)
            dy = gy[:, None, None] - (cc.float() + 0.5)
            valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            if not bool(valid.any()):
                continue
            lin = (rr * W + cc)[valid]
            dist2 = (dx * dx + dy * dy)[valid]
            tx, ty = dx[valid].clamp(-clip, clip), dy[valid].clamp(-clip, clip)
            o1 = torch.argsort(dist2, stable=True)
            o2 = torch.argsort(lin[o1], stable=True)
            order = o1[o2]
            ls = lin[order]
            first = torch.ones_like(ls, dtype=torch.bool)
            first[1:] = ls[1:] != ls[:-1]
            sel = order[first]
            pred = dense_offset[b].reshape(2, -1)[:, lin[sel]]
            total = total + (pred[0] - tx[sel]).abs().sum() + \
                (pred[1] - ty[sel]).abs().sum()
            count += int(sel.numel())
        return total / max(count, 1)

    def loss_by_feat(self, preds_dicts, batch_gt_instances_3d, *args,
                     **kwargs):
        loss_dict = super().loss_by_feat(preds_dicts, batch_gt_instances_3d,
                                         *args, **kwargs)
        if self.query_init == 'offset':
            loss_dict['loss_offset'] = self.loss_offset_weight * \
                self.offset_l1(preds_dicts[0][0]['dense_offset'].float(),
                               batch_gt_instances_3d)
        return loss_dict
