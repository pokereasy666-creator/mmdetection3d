# -*- coding: utf-8 -*-
"""Dev-side CPU tests for the exec doc #3 v1.1 model code (projects/QueryLoc)
against the UNCHANGED projects/BEVFusion TransFusionHead (see
ql_stub_env.py for what is a stand-in). Run from the repo root:
  python projects/DomeDETR/tests/test_exec3_model_cpu.py [--report PATH]
"""
import copy
import json
import os.path as osp
import sys
import warnings

import torch

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import ql_stub_env as Q  # noqa: E402
import test_query_diag_cpu as T  # noqa: E402

warnings.filterwarnings('ignore')
case = T.case
E = Q.install()
SC = E['subcell']
REPORT = []


def build(**over):
    return E['MODELS'].build(Q.head_cfg(**over))


def pair(**over):
    """Original 1-layer head and a QueryLoc head with the same weights."""
    torch.manual_seed(0)
    orig = build()
    ql = build(type='QueryLocTransFusionHead', **over)
    ql.load_state_dict(orig.state_dict(), strict=False)
    return orig, ql


def feats(seed=0, B=2):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, 16, Q.GRID, Q.GRID, generator=g)


GTS = [Q.make_gt(7, 11), Q.make_gt(5, 12)]


def losses(head, x, gts=GTS, seed=7):
    head.train()
    torch.manual_seed(seed)
    return head.loss_by_feat(head([x], None), gts)


@case('3m bilinear_gather: exact on cell centres (fp32 / fp16), bilinear '
      'off-centre, gradient to features only')
def test_bilinear(tmp):
    B, C, H, W = 2, 5, 7, 6
    f = torch.randn(B, C, H * W, requires_grad=True)
    r = torch.randint(0, H, (B, 9))
    c = torch.randint(0, W, (B, 9))
    pos = torch.stack([r.float() + 0.5, c.float() + 0.5], -1)
    ref = f.gather(2, (r * W + c)[:, None, :].expand(-1, C, -1))
    assert torch.equal(SC.bilinear_gather(f, pos, H, W), ref)
    fh = f.detach().half()
    refh = fh.gather(2, (r * W + c)[:, None, :].expand(-1, C, -1))
    out = SC.bilinear_gather(fh, pos, H, W)
    assert out.dtype == torch.float16 and torch.equal(out, refh)
    p2 = torch.tensor([[[2.75, 1.5]]])                  # row 2.25, col 1.0
    g = torch.arange(H * W, dtype=torch.float32).view(1, 1, -1)
    v = float(SC.bilinear_gather(g, p2, H, W))
    want = 0.75 * (2 * W + 1) + 0.25 * (3 * W + 1)
    assert abs(v - want) < 1e-5, (v, want)
    pp = pos.clone().requires_grad_(True)
    SC.bilinear_gather(f, pp + 0.3, H, W).sum().backward()
    assert f.grad is not None and float(f.grad.abs().sum()) > 0
    assert pp.grad is None


@case('3m S-M1 / S-M0 (CPU): zero-offset M-b (kept by init_weights) == '
      'original head: outputs, all losses except loss_offset, features '
      'bit-identical')
def test_mb_zero(tmp):
    orig, ql = pair(query_init='offset')
    ql.init_weights()               # runner order: init_weights, then load
    ql.load_state_dict(orig.state_dict(), strict=False)
    assert not ql.offset_head[-1].weight.any() and \
        not ql.offset_head[-1].bias.any()
    x = feats(1)
    orig.eval()
    ql.eval()
    ql.debug_check_gather = True
    with torch.no_grad():
        a, b = orig([x], None)[0][0], ql([x], None)[0][0]
    for k in a:
        assert torch.equal(a[k], b[k]), k
    assert torch.equal(orig.query_labels, ql.query_labels)
    assert ql.debug_gather_equal == (True, True), ql.debug_gather_equal
    la, lb = losses(orig, x), losses(ql, x)
    assert set(lb) == set(la) | {'loss_offset'}, (sorted(la), sorted(lb))
    for k in la:
        assert torch.equal(la[k], lb[k]), (k, la[k], lb[k])
    assert float(lb['loss_offset']) > 0
    REPORT.append('S-M1 (CPU): zero-offset M-b vs original head, train mode, '
                  'same seed: %s identical; loss_offset %.4f'
                  % (', '.join(sorted(la)), float(lb['loss_offset'])))


@case('3m L = 1: QueryLoc get_targets / losses == original (cell mode)')
def test_l1_targets(tmp):
    orig, ql = pair(query_init='cell')
    x = feats(2)
    orig.train()
    ql.train()
    torch.manual_seed(3)
    pa = orig([x], None)
    torch.manual_seed(3)
    pb = ql([x], None)
    ta = orig.get_targets(GTS, pa[0])
    tb = ql.get_targets(GTS, pb[0])
    for i, (u, v) in enumerate(zip(ta, tb)):
        if torch.is_tensor(u):
            assert torch.equal(u, v), i
        else:
            assert u == v, (i, u, v)
    la, lb = losses(orig, x), losses(ql, x)
    for k in la:
        assert torch.equal(la[k], lb[k]), k


def layer_preds(P, L, B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = P * L
    d = {'center': torch.rand(B, 2, n, generator=g) * Q.GRID,
         'height': torch.randn(B, 1, n, generator=g) * 0.3 - 0.3,
         'dim': torch.randn(B, 3, n, generator=g) * 0.3 + 0.5,
         'rot': torch.randn(B, 2, n, generator=g),
         'vel': torch.randn(B, 2, n, generator=g),
         'heatmap': torch.randn(B, 10, n, generator=g),
         'query_heatmap_score': torch.rand(B, 10, P, generator=g),
         'dense_heatmap': torch.randn(B, 10, Q.GRID, Q.GRID, generator=g)}
    return d


@case('3m S-D3L: 3-layer per-layer targets == single-layer targets; losses '
      'equal up to the total-positives normalization of the original code')
def test_d3l(tmp):
    torch.manual_seed(0)
    P, L = 20, 3
    d3 = build(type='QueryLocTransFusionHead', num_decoder_layers=L,
               query_init='cell')
    one = build()                       # original head, 1 layer
    preds = layer_preds(P, L)
    # loss_by_feat applies sigmoid_ IN PLACE to dense_heatmap (original
    # clip_sigmoid), so every call gets its own copy of the predictions
    t3 = d3.get_targets(GTS, [copy.deepcopy(preds)])
    l3 = d3.loss_by_feat(([copy.deepcopy(preds)], ), GTS)
    rows = []
    for i in range(L):
        sl = {k: (v[..., i * P:(i + 1) * P]
                  if k not in ('query_heatmap_score', 'dense_heatmap')
                  else v) for k, v in preds.items()}
        t1 = one.get_targets(GTS, [copy.deepcopy(sl)])
        l1 = one.loss_by_feat(([copy.deepcopy(sl)], ), GTS)
        assert torch.equal(t3[0][:, i * P:(i + 1) * P], t1[0])   # labels
        assert torch.equal(t3[1][:, i * P:(i + 1) * P], t1[1])   # weights
        assert torch.equal(t3[2][:, i * P:(i + 1) * P], t1[2])   # targets
        assert torch.equal(t3[3][:, i * P:(i + 1) * P], t1[3])
        assert torch.equal(t3[7], t1[7])                          # heatmap
        n_i = int(t1[5])
        pre = 'layer_-1' if i == L - 1 else 'layer_%d' % i
        for kind in ('loss_cls', 'loss_bbox'):
            v3 = float(l3['%s_%s' % (pre, kind)])
            v1 = float(l1['layer_-1_%s' % kind])
            lhs, rhs = v3 * max(int(t3[5]), 1), v1 * max(n_i, 1)
            assert abs(lhs - rhs) <= 1e-5 * max(abs(rhs), 1e-6), \
                (i, kind, lhs, rhs)
            rows.append((i, kind, v3, v1, n_i))
        assert torch.equal(l3['loss_heatmap'], l1['loss_heatmap'])
    num_tot = int(t3[5])
    assert num_tot == sum(r[4] for r in rows) // 2
    REPORT.append('S-D3L (CPU): per-layer labels / label weights / bbox '
                  'targets / bbox weights and heatmap target identical to '
                  'the single-layer original; positives per layer %s, total '
                  '%d' % ([r[4] for r in rows[::2]], num_tot))
    for i, kind, v3, v1, n_i in rows:
        REPORT.append('  layer %d %-9s 3-layer %.6f  single %.6f  ratio '
                      '%.4f = n_layer / n_total %d/%d'
                      % (i, kind, v3, v1, v3 / v1, n_i, num_tot))
    REPORT.append('  -> literal equality fails by the factor n_layer / '
                  'n_total (original code: avg_factor = positives of all '
                  'layers); equality holds after removing it.')


@case('3m D3 copy init: layers 1-2 copied from layer 0 on load (mmengine '
      'and torch loaders); a 3-layer state dict is not overwritten')
def test_copy_init(tmp):
    from mmengine.runner.checkpoint import load_state_dict
    torch.manual_seed(0)
    one = build()
    one.prediction_heads[0].center[0].bn.running_mean.fill_(0.37)
    sd = {'bbox_head.' + k: v for k, v in one.state_dict().items()}

    class Wrap(torch.nn.Module):

        def __init__(self, h):
            super().__init__()
            self.bbox_head = h

    for loader in ('mmengine', 'torch'):
        d3 = build(type='QueryLocTransFusionHead', num_decoder_layers=3)
        d3.init_weights()           # runner order: init_weights, then load
        w = Wrap(d3)
        if loader == 'mmengine':
            load_state_dict(w, copy.deepcopy(sd), strict=False)
        else:
            w.load_state_dict(copy.deepcopy(sd), strict=False)
        for sub in ('decoder', 'prediction_heads'):
            s0 = getattr(d3, sub)[0].state_dict()
            for i in (1, 2):
                si = getattr(d3, sub)[i].state_dict()
                assert set(s0) == set(si)
                for k in s0:
                    assert torch.equal(s0[k], si[k]), (loader, sub, i, k)
        assert sorted(d3.copied_layers) == sorted(
            'bbox_head.%s.%d.' % (s, i) for s in ('decoder',
                                                  'prediction_heads')
            for i in (1, 2)), d3.copied_layers
        assert torch.equal(d3.prediction_heads[2].center[0].bn.running_mean,
                           one.prediction_heads[0].center[0].bn.running_mean)
    # rehearse.py start-state check (S-T2) on the loaded head and on a head
    # whose copy did not happen
    sys.path.insert(0, osp.join(Q.REPO, 'projects/QueryLoc/tools'))
    import rehearse  # noqa: E402
    assert all(rehearse.start_state(d3).values()) and \
        len(rehearse.start_state(d3)) == 4
    raw = build(type='QueryLocTransFusionHead', num_decoder_layers=3)
    raw.init_weights()
    assert not all(rehearse.start_state(raw).values())
    mb = build(type='QueryLocTransFusionHead', query_init='offset')
    mb.init_weights()
    assert rehearse.start_state(mb) == {
        'offset_head last conv all zero': True}
    assert rehearse.start_state(build()) == {}
    d3b = build(type='QueryLocTransFusionHead', num_decoder_layers=3)
    with torch.no_grad():
        d3b.decoder[2].norms[0].weight.fill_(5.0)
    sd3 = {'bbox_head.' + k: v for k, v in d3b.state_dict().items()}
    d3c = build(type='QueryLocTransFusionHead', num_decoder_layers=3)
    load_state_dict(Wrap(d3c), sd3, strict=False)
    assert d3c.copied_layers == [] and \
        float(d3c.decoder[2].norms[0].weight[0]) == 5.0


@case('3m M-a target: GT on cell centres == original target (up to the '
      'float32 rounding of the GT grid coordinate, positives identical); '
      'sub-cell GT skews the Gaussian toward the true centre')
def test_ma_target(tmp):
    orig, ma = pair(query_init='dark', unbiased_heatmap=True)
    x = feats(4)
    for seed in (21, 22, 23):
        gt = Q.make_gt(9, seed, cell_centres=True)
        preds = orig([x], None)[0]
        sl = {k: v[0:1] for k, v in preds[0].items()}
        a = orig.get_targets_single(gt, sl, 0)[7]
        b = ma.get_targets_single(gt, sl, 0)[7]
        assert float((a - b).abs().max()) < 1e-6, float((a - b).abs().max())
        assert torch.equal(a.eq(1), b.eq(1)) and torch.equal(a > 0, b > 0)
    # one GT at a known sub-cell position
    from mmengine.structures import InstanceData
    t = torch.zeros(1, 9)
    t[0, 0] = (10 + 0.85) * 0.6 - Q.PC      # row 10, 0.35 cell towards +x
    t[0, 1] = (6 + 0.30) * 0.6 - Q.PC       # col 6, 0.20 cell towards -y
    t[0, 3:6] = torch.tensor([2.0, 1.0, 1.5])
    gt = InstanceData(bboxes_3d=Q.Boxes(t), labels_3d=torch.tensor([3]))
    sl = {k: v[0:1] for k, v in orig([x], None)[0][0].items()}
    hm_o = orig.get_targets_single(gt, sl, 0)[7][0, 3]
    hm_u = ma.get_targets_single(gt, sl, 0)[7][0, 3]
    assert float(hm_u[10, 6]) == 1.0 and int(hm_u.eq(1).sum()) == 1
    assert int(hm_o.eq(1).sum()) == 1
    assert float(hm_u[11, 6]) > float(hm_u[9, 6])     # +x side heavier
    assert float(hm_u[10, 5]) > float(hm_u[10, 7])    # -y side heavier
    assert abs(float(hm_o[11, 6]) - float(hm_o[9, 6])) < 1e-7
    assert torch.equal((hm_u > 0), (hm_o > 0))        # same window


@case('3m M-b dense L1: hand-computed targets, nearest-GT conflict rule, '
      'delta detached (detection losses do not reach the offset head)')
def test_mb_loss(tmp):
    orig, mb = pair(query_init='offset')
    from mmengine.structures import InstanceData
    t = torch.zeros(2, 9)
    t[:, 3:6] = torch.tensor([2.0, 1.0, 1.5])
    t[0, 0], t[0, 1] = (5 + 0.80) * 0.6 - Q.PC, (5 + 0.50) * 0.6 - Q.PC
    t[1, 0], t[1, 1] = (8 + 0.20) * 0.6 - Q.PC, (5 + 0.50) * 0.6 - Q.PC
    gt = InstanceData(bboxes_3d=Q.Boxes(t), labels_3d=torch.tensor([0, 1]))
    zero = torch.zeros(1, 2, Q.GRID, Q.GRID)
    got = float(mb.offset_l1(zero, [gt]))
    gx = torch.tensor([5.8, 8.2])
    gy = torch.tensor([5.5, 5.5])
    tot, n = 0.0, 0
    for r in range(Q.GRID):
        for c in range(Q.GRID):
            cand = [j for j in range(2)
                    if abs(r - int(gx[j])) <= 3 and abs(c - int(gy[j])) <= 3]
            if not cand:
                continue
            j = min(cand, key=lambda j: (float(gx[j]) - r - 0.5) ** 2 +
                    (float(gy[j]) - c - 0.5) ** 2)
            tx = max(-3.0, min(3.0, float(gx[j]) - r - 0.5))
            ty = max(-3.0, min(3.0, float(gy[j]) - c - 0.5))
            tot += abs(tx) + abs(ty)
            n += 1
    want = tot / n
    assert n == 7 * 7 + 3 * 7 and abs(got - want) < 1e-5, (got, want, n)
    with torch.no_grad():
        mb.offset_head[-1].bias.copy_(torch.tensor([0.3, -0.2]))
    x = feats(5)
    mb.train()
    torch.manual_seed(9)
    out = mb([x], None)
    det = sum(v for k, v in mb.loss_by_feat(out, GTS).items()
              if 'loss' in k and k != 'loss_offset')
    mb.zero_grad()
    det.backward(retain_graph=True)
    g_det = [p.grad for p in mb.offset_head.parameters()]
    assert all(g is None or float(g.abs().sum()) == 0 for g in g_det)
    mb.zero_grad()
    mb.loss_by_feat(out, GTS)['loss_offset'].backward()
    assert float(mb.offset_head[-1].bias.grad.abs().sum()) > 0
    d = float(torch.tanh(torch.tensor(0.3)) * 3)
    cell = mb.bev_pos[0].gather(0, torch.zeros(1, 2, dtype=torch.long))
    assert mb.query_pos_init.shape[-1] == 2 and cell is not None
    diff = (mb.query_pos_init[..., 0] - mb.query_pos_init[..., 0].floor())
    assert torch.allclose(diff, torch.full_like(diff, (0.5 + d) % 1.0),
                          atol=1e-5), diff[0, :3]


@case('3m S-T5 (CPU): top-K cells, labels and query_heatmap_score of '
      'M-a (DARK) and M-b (non-zero offsets) == original; DARK positions '
      '== subcell.dark_offsets')
def test_st5(tmp):
    x = feats(6)
    for over in (dict(query_init='dark'), dict(query_init='offset')):
        orig, ql = pair(**over)
        if over['query_init'] == 'offset':
            with torch.no_grad():
                ql.offset_head[-1].bias.copy_(torch.tensor([0.4, -0.1]))
        orig.eval()
        ql.eval()
        with torch.no_grad():
            a, b = orig([x], None)[0][0], ql([x], None)[0][0]
        assert torch.equal(orig.query_labels, ql.query_labels)
        assert torch.equal(a['query_heatmap_score'], b['query_heatmap_score'])
        assert torch.equal(a['dense_heatmap'], b['dense_heatmap'])
        assert not torch.equal(a['center'], b['center'])   # positions moved
    orig, ma = pair(query_init='dark')
    ma.eval()
    with torch.no_grad():
        ma([x], None)
        dh = ma.heatmap_head(ma.shared_conv(x).float()).sigmoid()
    from projects.QueryLoc.queryloc.subcell import (dark_offsets,
                                                    neighborhood3x3)
    hm = dh.clone()                                 # replicate top-K
    pad = 1
    lm = torch.zeros_like(hm)
    lm[:, :, pad:-pad, pad:-pad] = torch.nn.functional.max_pool2d(hm, 3, 1)
    lm[:, 8], lm[:, 9] = hm[:, 8], hm[:, 9]
    top = (hm * (hm == lm)).view(2, -1).argsort(-1, descending=True)[:, :20]
    idx, cls = top % (Q.GRID * Q.GRID), top // (Q.GRID * Q.GRID)
    nb, _, border = neighborhood3x3(dh, idx, cls)
    off, _, _ = dark_offsets(nb, border)
    assert torch.equal(cls, ma.query_labels)
    cellp = torch.stack([(idx // Q.GRID).float() + 0.5,
                         (idx % Q.GRID).float() + 0.5], -1)
    assert torch.allclose(ma.query_pos_init, cellp + off, atol=1e-6)


@case('3m freezing: requires_grad off + eval kept through train() and a '
      'SyncBN-like module swap; trainable set = fuser, BEV backbone/neck, '
      'head')
def test_freeze(tmp):
    M = E['QueryLocBEVFusion']
    m = M()
    frozen = ('img_backbone', 'img_neck', 'view_transform',
              'pts_voxel_layer', 'pts_voxel_encoder', 'pts_middle_encoder')
    for n in frozen:
        assert all(not p.requires_grad for p in getattr(m, n).parameters())
    for n in ('fusion_layer', 'pts_backbone', 'pts_neck', 'bbox_head'):
        assert all(p.requires_grad for p in getattr(m, n).parameters())
    m.train()
    for n in frozen:
        assert not getattr(m, n).training, n
    assert m.fusion_layer.training and m.bbox_head.training
    bn = m.img_neck[1]
    new = torch.nn.BatchNorm2d(4)
    new.weight, new.bias = bn.weight, bn.bias       # as convert_sync_bn
    m.img_neck[1] = new
    m.train()
    assert not m.img_neck[1].training
    assert not m.img_neck[1].weight.requires_grad
    try:
        E['model_mod'].freeze_modules(m, ['no_such_module'])
        raise AssertionError('typo did not raise')
    except AttributeError:
        pass
    n_frozen = sum(p.numel() for n in frozen
                   for p in getattr(m, n).parameters())
    assert m.num_frozen_params == n_frozen


def main():
    tmp = None
    test_bilinear(tmp)
    test_mb_zero(tmp)
    test_l1_targets(tmp)
    test_d3l(tmp)
    test_copy_init(tmp)
    test_ma_target(tmp)
    test_mb_loss(tmp)
    test_st5(tmp)
    test_freeze(tmp)
    n_fail = sum(1 for r in T.RESULTS if r[1] != 'PASS')
    print('\n'.join(REPORT))
    print('\n%d tests, %d failed' % (len(T.RESULTS), n_fail))
    if '--report' in sys.argv:
        path = sys.argv[sys.argv.index('--report') + 1]
        with open(path, 'w') as f:
            json.dump({'results': T.RESULTS, 'report': REPORT}, f, indent=2)
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
