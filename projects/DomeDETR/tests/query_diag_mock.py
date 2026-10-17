# -*- coding: utf-8 -*-
"""CPU mock of BEVFusion's TransFusionHead for testing tools/query_diag.

Dev-side only (not deployed to the server). The query initialization,
predict_by_feat score computation and TransFusionBBoxCoder.decode below are
copied from projects/BEVFusion (transfusion_head.py:227-262, 298-304,
358-386; utils.py:72-98), INCLUDING the in-place rewrite of 'center' by
decode, which is what finding F1 is about. Only the backbone and the
transformer decoder are replaced by synthetic stand-ins.

Child-process entry points (run by test_query_diag_cpu.py):
  python query_diag_mock.py recall <impl_dir> <out_dir> <cfg> <ckpt>
  python query_diag_mock.py oracle <impl_dir> <work_dir> <cfg> <ckpt> <tag>
                                   [diag_oracle_query args ...]
"""
import importlib.util
import json
import math
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

CLASSES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
           'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
TEST_CFG = dict(dataset='nuScenes', grid_size=[1440, 1440, 41],
                out_size_factor=8, voxel_size=[0.075, 0.075],
                pc_range=[-54.0, -54.0], nms_type=None)


class MockBBoxCoder:
    """utils.py TransFusionBBoxCoder.decode, verbatim in-place updates."""

    def __init__(self):
        self.pc_range = [-54.0, -54.0]
        self.out_size_factor = 8
        self.voxel_size = [0.075, 0.075]

    def decode(self, heatmap, rot, dim, center, height, vel, filter=False):
        final_preds = heatmap.max(1, keepdims=False).indices
        final_scores = heatmap.max(1, keepdims=False).values
        center[:,
               0, :] = center[:,
                              0, :] * self.out_size_factor * self.voxel_size[
                                  0] + self.pc_range[0]
        center[:,
               1, :] = center[:,
                              1, :] * self.out_size_factor * self.voxel_size[
                                  1] + self.pc_range[1]
        dim[:, 0, :] = dim[:, 0, :].exp()
        dim[:, 1, :] = dim[:, 1, :].exp()
        dim[:, 2, :] = dim[:, 2, :].exp()
        height = height - dim[:, 2:3, :] * 0.5
        rots, rotc = rot[:, 0:1, :], rot[:, 1:2, :]
        rot = torch.atan2(rots, rotc)
        final_box_preds = torch.cat([center, height, dim, rot, vel],
                                    dim=1).permute(0, 2, 1)
        return [{'bboxes': final_box_preds[i], 'scores': final_scores[i],
                 'labels': final_preds[i]} for i in range(heatmap.shape[0])]


class LogitSource(nn.Module):
    """Stands in for heatmap_head: returns logits staged by the detector."""

    def __init__(self):
        super().__init__()
        self.staged = None

    def forward(self, x):
        return self.staged.clone()


class MockHead(nn.Module):

    def __init__(self, num_proposals=200):
        super().__init__()
        self.num_classes = 10
        self.num_proposals = num_proposals
        self.nms_kernel_size = 3
        self.test_cfg = dict(TEST_CFG)
        self.bbox_coder = MockBBoxCoder()
        self.heatmap_head = LogitSource()
        x_size = self.test_cfg['grid_size'][0] // self.test_cfg[
            'out_size_factor']
        y_size = self.test_cfg['grid_size'][1] // self.test_cfg[
            'out_size_factor']
        self.bev_pos = self.create_2D_grid(x_size, y_size)
        self.decoder_offsets = None   # [B, 2, H*W] grid-unit offsets

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid],
            indexing='ij')
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base

    def forward_single(self, inputs, metas):
        batch_size = inputs.shape[0]
        fusion_feat = inputs
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(fusion_feat.device)
        dense_heatmap = self.heatmap_head(fusion_feat.float())
        heatmap = dense_heatmap.detach().sigmoid()
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        local_max_inner = F.max_pool2d(
            heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
        local_max[:, :, padding:(-padding),
                  padding:(-padding)] = local_max_inner
        if self.test_cfg['dataset'] == 'nuScenes':
            local_max[:, 8, ] = F.max_pool2d(
                heatmap[:, 8], kernel_size=1, stride=1, padding=0)
            local_max[:, 9, ] = F.max_pool2d(
                heatmap[:, 9], kernel_size=1, stride=1, padding=0)
        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)
        top_proposals = heatmap.view(batch_size, -1).argsort(
            dim=-1, descending=True)[..., :self.num_proposals]
        top_proposals_class = top_proposals // heatmap.shape[-1]
        top_proposals_index = top_proposals % heatmap.shape[-1]
        self.query_labels = top_proposals_class
        query_pos = bev_pos.gather(
            index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(
                -1, -1, bev_pos.shape[-1]),
            dim=1,
        )
        # Decoder stand-in: per-cell offset (grid units) toward the GT.
        off = self.decoder_offsets.gather(
            2, top_proposals_index[:, None, :].expand(-1, 2, -1))
        P = self.num_proposals
        res_layer = {
            'center': off + query_pos.permute(0, 2, 1),
            'height': torch.zeros(batch_size, 1, P),
            'dim': torch.zeros(batch_size, 3, P),
            'rot': torch.cat([torch.zeros(batch_size, 1, P),
                              torch.ones(batch_size, 1, P)], 1),
            'vel': torch.zeros(batch_size, 2, P),
            'heatmap': torch.full((batch_size, self.num_classes, P), 2.0),
        }
        ret_dicts = [res_layer]
        ret_dicts[0]['query_heatmap_score'] = heatmap.gather(
            index=top_proposals_index[:,
                                      None, :].expand(-1, self.num_classes,
                                                      -1),
            dim=-1,
        )
        ret_dicts[0]['dense_heatmap'] = dense_heatmap
        new_res = {}
        for key in ret_dicts[0].keys():
            if key not in [
                    'dense_heatmap', 'dense_heatmap_old', 'query_heatmap_score'
            ]:
                new_res[key] = torch.cat(
                    [ret_dict[key] for ret_dict in ret_dicts], dim=-1)
            else:
                new_res[key] = ret_dicts[0][key]
        return [new_res]

    def forward(self, feats, metas):
        return ([self.forward_single(feats, metas)[0]], )

    def predict(self, batch_feats, batch_input_metas):
        preds_dicts = self(batch_feats, batch_input_metas)
        return self.predict_by_feat(preds_dicts, batch_input_metas)

    def predict_by_feat(self, preds_dicts, metas):
        rets = []
        for layer_id, preds_dict in enumerate(preds_dicts):
            batch_score = preds_dict[0]['heatmap'][
                ..., -self.num_proposals:].sigmoid()
            one_hot = F.one_hot(
                self.query_labels,
                num_classes=self.num_classes).permute(0, 2, 1)
            batch_score = batch_score * preds_dict[0][
                'query_heatmap_score'] * one_hot
            batch_center = preds_dict[0]['center'][..., -self.num_proposals:]
            batch_height = preds_dict[0]['height'][..., -self.num_proposals:]
            batch_dim = preds_dict[0]['dim'][..., -self.num_proposals:]
            batch_rot = preds_dict[0]['rot'][..., -self.num_proposals:]
            batch_vel = preds_dict[0]['vel'][..., -self.num_proposals:]
            temp = self.bbox_coder.decode(batch_score, batch_rot, batch_dim,
                                          batch_center, batch_height,
                                          batch_vel, filter=True)
            rets.append(temp)
        return rets[0]


class Boxes:

    def __init__(self, t):
        self.tensor = t


class Sample:
    """Minimal Det3DDataSample stand-in (metainfo + eval_ann_info)."""

    def __init__(self, gt, labels):
        self.metainfo = {}
        self.eval_ann_info = {'gt_bboxes_3d': Boxes(gt),
                              'gt_labels_3d': labels}


def cell_of(x, y):
    return (int(math.floor((x + 54.0) / 0.6)),
            int(math.floor((y + 54.0) / 0.6)))


def make_frame(rng, n_gt=45, n_distract=320):
    """Synthetic GT + staged logits + decoder offsets for one frame."""
    H = W = 180
    lab, xs, ys = [], [], []
    while len(lab) < n_gt:
        r = 50.0 * math.sqrt(rng.uniform())
        a = rng.uniform(0, 2 * math.pi)
        x, y = r * math.cos(a), r * math.sin(a)
        lab.append(int(rng.integers(0, 10)))
        xs.append(x)
        ys.append(y)
    lab = np.asarray(lab, dtype=np.int64)
    lab[:3] = -1                                        # ignored classes
    gt = torch.zeros(n_gt, 9)
    gt[:, 0] = torch.tensor(xs)
    gt[:, 1] = torch.tensor(ys)
    logits = torch.full((10, H, W), -9.0) + \
        torch.from_numpy(rng.normal(0, 0.3, (10, H, W))).float()
    off = torch.zeros(2, H * W)
    for i in range(n_gt):
        if lab[i] < 0:
            continue
        d = math.hypot(xs[i], ys[i])
        if rng.uniform() < 0.05 or (d > 35 and rng.uniform() < 0.35):
            continue                                    # no response at all
        r, c = cell_of(xs[i], ys[i])
        if not (0 <= r < H and 0 <= c < W):
            continue
        logits[lab[i], r, c] = 3.0 - 0.1 * d + rng.normal(0, 0.5)
        # decoder pulls the query from the cell center to the GT center
        gx = (xs[i] + 54.0) / 0.6
        gy = (ys[i] + 54.0) / 0.6
        off[0, r * W + c] = gx - (r + 0.5)
        off[1, r * W + c] = gy - (c + 0.5)
    for _ in range(n_distract):
        ch, r, c = (int(rng.integers(0, 10)), int(rng.integers(1, H - 1)),
                    int(rng.integers(1, W - 1)))
        logits[ch, r, c] = max(float(logits[ch, r, c]),
                               rng.uniform(-3.0, 0.5))
    return gt, lab, logits, off


class MockDetector(nn.Module):

    def __init__(self):
        super().__init__()
        self.bbox_head = MockHead()

    def cuda(self, *a, **k):          # CPU-only test environment
        return self

    def test_step(self, data):
        return self.forward(data['inputs'], data['data_samples'],
                            mode='predict')

    def forward(self, inputs, data_samples, mode='predict'):
        return self.predict(inputs, data_samples)

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        head = self.bbox_head
        head.heatmap_head.staged = batch_inputs_dict['logits']
        head.decoder_offsets = batch_inputs_dict['offsets']
        feats = torch.zeros(batch_inputs_dict['logits'].shape[0], 4, 180, 180)
        return head.predict(feats, [s.metainfo for s in batch_data_samples])


class Loader(list):

    class _DS:
        metainfo = {'classes': tuple(CLASSES)}

    dataset = _DS()


def make_loader(n_frames, batch, seed):
    rng = np.random.default_rng(seed)
    frames = [make_frame(rng) for _ in range(n_frames)]
    out = Loader()
    for i in range(0, n_frames, batch):
        fr = frames[i:i + batch]
        out.append({'inputs': {'logits': torch.stack([f[2] for f in fr]),
                               'offsets': torch.stack([f[3] for f in fr])},
                    'data_samples': [Sample(f[0], f[1]) for f in fr]})
    return out


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def patch_mmengine(loader, runner_test=None):
    from mmengine.config import Config
    from mmengine.runner import Runner

    def fromfile(path, *a, **k):
        return Config(dict(test_dataloader=dict(dataset=dict(
            metainfo=dict(classes=CLASSES))),
            val_cfg=dict(), test_cfg=dict(), class_names=CLASSES))

    class FakeRunner:

        def __init__(self, cfg):
            self.cfg = cfg
            self.model = MockDetector()

        def load_checkpoint(self, *a, **k):
            return {}

        def build_dataloader(self, cfg):
            return loader

        def test(self):
            return runner_test(self)

    Config.fromfile = staticmethod(fromfile)
    Runner.from_cfg = classmethod(lambda cls, cfg: FakeRunner(cfg))


def child_recall(impl_dir, out_dir, cfg, ckpt):
    loader = make_loader(24, 2, seed=0)
    patch_mmengine(loader)
    sys.path.insert(0, impl_dir)
    mod = load_module('diag_query_recall',
                      osp.join(impl_dir, 'diag_query_recall.py'))
    sys.argv = ['diag_query_recall.py', cfg, ckpt, '--out-dir', out_dir]
    mod.main()


def child_oracle(impl_dir, work_dir, cfg, ckpt, tag, extra):
    loader = make_loader(12, 2, seed=1)
    checks = {'frames': 0, 'restored_ok': 0, 'restored_bad': 0,
              'injected_selected': 0}

    def runner_test(runner):
        dc = sys.modules['diag_common']
        model = runner.model
        hit = tot = 0
        with torch.no_grad():
            for data in loader:
                res = model.test_step(data)
                S = dc.STATE
                checks['frames'] += len(data['data_samples'])
                if 'oracle_mask' in S:
                    # score restoration: injected queries must carry the
                    # ORIGINAL pre-NMS sigmoid of their (class, cell)
                    head = model.bbox_head
                    d = dc.first_dict(S['head_out'])
                    idx, cls, _ = dc.replicate_init(head, S['logits'],
                                                    head.num_proposals)
                    B, C, H, W = S['logits'].shape
                    key = cls * H * W + idx
                    is_or = S['oracle_mask'].gather(1, key)
                    raw = S['logits_raw'].sigmoid().reshape(B, -1).gather(
                        1, key)
                    q = d['query_heatmap_score'].gather(
                        1, cls[:, None, :]).squeeze(1)
                    ok = torch.isclose(q[is_or], raw[is_or])
                    checks['restored_ok'] += int(ok.sum())
                    checks['restored_bad'] += int((~ok).sum())
                    checks['injected_selected'] += int(is_or.sum())
                # proxy metric: GT with a positive-score box within 2 m
                for b, s in enumerate(data['data_samples']):
                    gt = s.eval_ann_info['gt_bboxes_3d'].tensor[:, :2]
                    lab = torch.as_tensor(s.eval_ann_info['gt_labels_3d'])
                    gt, lab = gt[lab >= 0], lab[lab >= 0]
                    bx = res[b]['bboxes'][:, :2]
                    keep = res[b]['scores'] > 0
                    dd = torch.cdist(gt.double(), bx[keep].double())
                    hit += int((dd.min(1).values <= 2.0).sum()) \
                        if keep.any() else 0
                    tot += len(gt)
        v = hit / max(tot, 1)
        return {'NuScenes metric/pred_instances_3d_NuScenes/mAP': v,
                'NuScenes metric/pred_instances_3d_NuScenes/NDS': v / 2 + 0.3}

    patch_mmengine(loader, runner_test)
    sys.path.insert(0, impl_dir)
    mod = load_module('diag_oracle_query',
                      osp.join(impl_dir, 'diag_oracle_query.py'))
    sys.argv = ['diag_oracle_query.py', cfg, ckpt, '--work-dir', work_dir,
                '--tag', tag] + extra
    mod.main()
    with open(osp.join(work_dir, 'harness_checks_%s.json' % tag), 'w') as f:
        json.dump(checks, f)


if __name__ == '__main__':
    if sys.argv[1] == 'recall':
        child_recall(*sys.argv[2:6])
    elif sys.argv[1] == 'oracle':
        child_oracle(*sys.argv[2:7], sys.argv[7:])
    else:
        sys.exit('unknown mode')
