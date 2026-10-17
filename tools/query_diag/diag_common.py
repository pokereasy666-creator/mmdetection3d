# -*- coding: utf-8 -*-
"""Shared helpers for the TransFusion-head query diagnostics.

Target code base: mmdet3d 1.4.0 + projects/BEVFusion (TransFusionHead).
All logic below only READS the head's attributes; the head source is not
modified. Query initialization is re-computed from the dense heatmap logits
with exactly the same local-max NMS + global top-K rule as
TransFusionHead.forward_single, and the result is cross-checked against
head.query_labels after every forward pass.

This file is ASCII-only on purpose (pre-sync gate).
"""
import numpy as np
import torch
import torch.nn.functional as F

# nuScenes official detection ranges (detection_cvpr_2019), meters
NUS_CLASS_RANGE = {
    'car': 50.0, 'truck': 50.0, 'bus': 50.0, 'trailer': 50.0,
    'construction_vehicle': 50.0, 'pedestrian': 40.0, 'motorcycle': 40.0,
    'bicycle': 40.0, 'traffic_cone': 30.0, 'barrier': 30.0,
}

# Per-forward state filled by the patches below; cleared at every predict().
STATE = {}
# Persistent user callbacks (NOT cleared per forward):
#   HOOKS['logits_modifier'](logits) -> logits      (oracle injection)
#   HOOKS['head_out_modifier'](head_out) -> None    (in-place score fix)
HOOKS = {}


def unwrap(model):
    return model.module if hasattr(model, 'module') else model


def get_head(model):
    m = unwrap(model)
    head = getattr(m, 'bbox_head', None)
    if head is None:
        raise RuntimeError('model has no bbox_head; is this BEVFusion?')
    for name in ('heatmap_head', 'bev_pos', 'num_proposals',
                 'nms_kernel_size', 'test_cfg'):
        if not hasattr(head, name):
            raise RuntimeError(
                'bbox_head has no attribute %r; this script expects the '
                'TransFusionHead of projects/BEVFusion (mmdet3d 1.4).' % name)
    return head


def head_geometry(head):
    """Return (out_size_factor, vx, vy, x0, y0) used by the head decoder."""
    tc = head.test_cfg
    try:
        osf = float(tc['out_size_factor'])
        vx, vy = float(tc['voxel_size'][0]), float(tc['voxel_size'][1])
        x0, y0 = float(tc['pc_range'][0]), float(tc['pc_range'][1])
    except (KeyError, TypeError):
        bc = head.bbox_coder
        osf = float(bc.out_size_factor)
        vx, vy = float(bc.voxel_size[0]), float(bc.voxel_size[1])
        x0, y0 = float(bc.pc_range[0]), float(bc.pc_range[1])
    return osf, vx, vy, x0, y0


def probe_bev_pos(head, H, W):
    """Check the flat-index convention of head.bev_pos.

    Returns False if bev_pos[0, r*W + c] == (r+0.5, c+0.5)  (row <-> x),
    True  if bev_pos[0, r*W + c] == (c+0.5, r+0.5)  (transposed).
    Raises if neither holds.
    """
    bp = head.bev_pos.detach().float().cpu()
    if bp.dim() != 3 or bp.shape[1] != H * W:
        raise RuntimeError('unexpected bev_pos shape %s for H=%d W=%d'
                           % (tuple(bp.shape), H, W))
    probes = [(0, 0), (0, W - 1), (H - 1, 0), (H // 3, W // 2 + 7)]
    ok_rc, ok_cr = True, True
    for r, c in probes:
        v = bp[0, r * W + c]
        ok_rc &= bool(torch.allclose(v, torch.tensor([r + 0.5, c + 0.5])))
        ok_cr &= bool(torch.allclose(v, torch.tensor([c + 0.5, r + 0.5])))
    if ok_rc:
        return False
    if ok_cr:
        return True
    raise RuntimeError('bev_pos convention not recognized; stop and check '
                       'create_2D_grid in transfusion_head.py')


def replicate_init(head, logits, kmax):
    """Re-compute query initialization exactly like forward_single.

    Args:
        logits: dense heatmap logits [B, C, H, W] (output of heatmap_head).
        kmax: number of ranked proposals to return (>= head.num_proposals).
    Returns:
        idx  [B, kmax] flat cell index (into H*W)
        cls  [B, kmax] class channel of each proposal
        sig  [B, C, H*W] sigmoid heatmap BEFORE local-max NMS
    """
    heat = logits.detach().float().sigmoid()
    B, C, H, W = heat.shape
    k = int(head.nms_kernel_size)
    pad = k // 2
    local_max = torch.zeros_like(heat)
    inner = F.max_pool2d(heat, kernel_size=k, stride=1, padding=0)
    if pad > 0:
        local_max[:, :, pad:-pad, pad:-pad] = inner
    else:
        local_max = inner
    ds = head.test_cfg.get('dataset', None) if hasattr(
        head.test_cfg, 'get') else head.test_cfg['dataset']
    if ds == 'nuScenes':      # pedestrian & traffic_cone: no local-max NMS
        local_max[:, 8] = heat[:, 8]
        local_max[:, 9] = heat[:, 9]
    elif ds == 'Waymo':       # pedestrian & cyclist
        local_max[:, 1] = heat[:, 1]
        local_max[:, 2] = heat[:, 2]
    heat_nms = heat * (heat == local_max)
    flat = heat_nms.view(B, -1)
    kmax = min(int(kmax), flat.shape[1])
    order = flat.argsort(dim=-1, descending=True)[:, :kmax]
    cls = order // (H * W)
    idx = order % (H * W)
    return idx, cls, heat.view(B, C, -1)


def cells_to_xy(head, idx):
    """Flat cell index -> BEV metric xy using head.bev_pos (grid units)."""
    osf, vx, vy, x0, y0 = head_geometry(head)
    bp = head.bev_pos.to(idx.device).float()[0]          # [H*W, 2]
    g = bp[idx]                                           # [..., 2]
    x = g[..., 0] * osf * vx + x0
    y = g[..., 1] * osf * vy + y0
    return torch.stack([x, y], dim=-1)


def grid_to_xy(head, center):
    """Decoder 'center' output [B, 2, N] (grid units) -> metric [B, N, 2]."""
    osf, vx, vy, x0, y0 = head_geometry(head)
    x = center[:, 0] * osf * vx + x0
    y = center[:, 1] * osf * vy + y0
    return torch.stack([x, y], dim=-1)


def xy_to_flat(head, xy, H, W, transposed):
    """Metric xy [N, 2] -> flat cell index, same rounding as get_targets."""
    osf, vx, vy, x0, y0 = head_geometry(head)
    gx = ((xy[:, 0] - x0) / vx / osf).floor().long()
    gy = ((xy[:, 1] - y0) / vy / osf).floor().long()
    if not transposed:
        r, c = gx, gy
    else:
        r, c = gy, gx
    valid = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    return r, c, valid


def first_dict(obj):
    while isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            raise RuntimeError('empty head output')
        obj = obj[0]
    if not isinstance(obj, dict):
        raise RuntimeError('cannot find prediction dict in head output')
    return obj


def gt_from_sample(sample, class_names, use_class_range=True):
    """Return (xy [G,2] float tensor, labels [G] long, ego_dist [G])."""
    ann = getattr(sample, 'eval_ann_info', None)
    if ann is not None:
        boxes, labels = ann['gt_bboxes_3d'], ann['gt_labels_3d']
    elif hasattr(sample, 'gt_instances_3d') and \
            'bboxes_3d' in sample.gt_instances_3d:
        boxes = sample.gt_instances_3d.bboxes_3d
        labels = sample.gt_instances_3d.labels_3d
    else:
        raise RuntimeError(
            'no GT in data_sample (eval_ann_info is None and no '
            'gt_instances_3d). Keep load_eval_anns=True (mmdet3d 1.x '
            'default) in the test dataset.')
    t = boxes.tensor if hasattr(boxes, 'tensor') else torch.as_tensor(boxes)
    xy = t[:, :2].float().cpu()
    if torch.is_tensor(labels):
        labels = labels.detach().long().cpu()
    else:
        labels = torch.as_tensor(np.asarray(labels)).long().cpu()
    keep = labels >= 0
    xy, labels = xy[keep], labels[keep]
    dist = torch.linalg.norm(xy, dim=1)
    if use_class_range and len(labels) > 0:
        rng = torch.tensor([NUS_CLASS_RANGE.get(n, 1e9)
                            for n in class_names])
        keep = dist <= rng[labels]
        xy, labels, dist = xy[keep], labels[keep], dist[keep]
    return xy, labels, dist


def install_capture(model):
    """Patch predict / heatmap_head / head.forward to record one forward.

    After model.test_step(data) the following are in STATE:
      samples, logits [B,C,H,W], head_out (raw head output),
      query_labels [B,P]
    """
    m = unwrap(model)
    head = get_head(m)

    orig_predict = m.predict

    def predict(batch_inputs_dict, batch_data_samples, **kwargs):
        STATE.clear()
        STATE['samples'] = batch_data_samples
        return orig_predict(batch_inputs_dict, batch_data_samples, **kwargs)

    m.predict = predict

    def hm_hook(module, inputs, output):
        hook_fn = HOOKS.get('logits_modifier', None)
        STATE['logits_raw'] = output.detach()
        if hook_fn is not None:
            output = hook_fn(output)
        STATE['logits'] = output.detach()
        return output

    head.heatmap_head.register_forward_hook(hm_hook)

    orig_forward = head.forward

    def forward(*args, **kwargs):
        out = orig_forward(*args, **kwargs)
        STATE['query_labels'] = head.query_labels.detach().clone()
        post = HOOKS.get('head_out_modifier', None)
        if post is not None:
            post(out)
        STATE['head_out'] = out
        return out

    head.forward = forward
    return head


# ---------------------------------------------------------------------------
# Pure tensor helpers (no mmengine dependency, unit-testable on CPU)
# ---------------------------------------------------------------------------

def get_class_names(cfg, dataset=None):
    if dataset is not None:
        mi = getattr(dataset, 'metainfo', None)
        if mi and 'classes' in mi:
            return list(mi['classes'])
    try:
        return list(cfg.test_dataloader.dataset.metainfo['classes'])
    except Exception:
        pass
    if 'class_names' in cfg:
        return list(cfg.class_names)
    raise RuntimeError('cannot determine class names from config')


def per_gt_hits(g_xy, g_lab, q_xy, q_cls, thr):
    """First rank (0-based) of a query within t meters of each GT.

    Args:
        g_xy [G,2], g_lab [G], q_xy [K,2] (rank order), q_cls [K]
        thr: list of distance thresholds (m)
    Returns:
        rank_agn, rank_ca: long [G, T]; value K means "no query within t".
        rank_ca only counts queries whose class == GT class (in the
        TransFusion head a query's class is fixed at initialization and
        the final score is masked to that class).
    """
    K = q_xy.shape[0]
    D = torch.cdist(g_xy.double(), q_xy.double())
    same = q_cls[None, :] == g_lab[:, None]
    ar = torch.arange(K, device=D.device).expand_as(D)
    miss = torch.full_like(ar, K)
    ra, rc = [], []
    for t in thr:
        w = D <= t
        ra.append(torch.where(w, ar, miss).min(1).values)
        rc.append(torch.where(w & same, ar, miss).min(1).values)
    return torch.stack(ra, 1), torch.stack(rc, 1)


def oracle_cells(head, logits_b, xy, lab, H, W, transposed, mode, cap,
                 jitter=0, gen=None):
    """Cells (channel, row, col) to boost for one frame.

    mode 'loc'    : channel = argmax of the ORIGINAL logits at that cell
                    (location oracle only; class is still the network's)
    mode 'loc_cls': channel = GT class (location + class oracle)
    """
    r, c, valid = xy_to_flat(head, xy, H, W, transposed)
    r, c, lab = r[valid], c[valid], lab[valid]
    if jitter > 0 and len(r) > 0:
        dr = torch.randint(-jitter, jitter + 1, r.shape, generator=gen)
        dc_ = torch.randint(-jitter, jitter + 1, c.shape, generator=gen)
        r = (r + dr).clamp(0, H - 1)
        c = (c + dc_).clamp(0, W - 1)
    if len(r) == 0:
        e = torch.zeros(0, dtype=torch.long)
        return e, e, e, 0
    dev = logits_b.device
    if mode == 'loc_cls':
        ch = lab.clone()
    elif mode == 'loc':
        ch = logits_b[:, r.to(dev), c.to(dev)].argmax(0).cpu()
    else:
        raise ValueError(mode)
    flat = ch * (H * W) + r * W + c
    flat, first = np.unique(flat.numpy(), return_index=True)
    order = np.sort(first)
    ch, r, c = ch[order], r[order], c[order]
    n_capped = 0
    if len(ch) > cap:
        n_capped = len(ch) - cap
        ch, r, c = ch[:cap], r[:cap], c[:cap]
    return ch, r, c, n_capped


def restore_oracle_scores(qhs, cls, idx, oracle_mask, sig_raw):
    """In-place: for queries sitting on an injected cell, replace the
    heatmap score (which is ~1 after boosting) by the ORIGINAL pre-NMS
    sigmoid value of that cell, so that confidence is not leaked.

    qhs        [B, C, P]  query_heatmap_score (modified in place)
    cls, idx   [B, P]     class / flat cell of each initial query
    oracle_mask[B, C*HW]  bool, True at injected (channel, cell)
    sig_raw    [B, C, HW] sigmoid of the unmodified logits
    Returns number of queries whose score was restored.
    """
    B, C, HW = sig_raw.shape
    key = cls * HW + idx
    is_or = oracle_mask.gather(1, key)
    orig = sig_raw.reshape(B, -1).gather(1, key).to(qhs.dtype)
    cur = qhs.gather(1, cls[:, None, :]).squeeze(1)
    new = torch.where(is_or, orig, cur)
    qhs.scatter_(1, cls[:, None, :], new[:, None, :])
    return int(is_or.sum().item())
