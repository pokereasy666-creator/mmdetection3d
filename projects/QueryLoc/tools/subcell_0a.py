#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exec doc #3 step 0a: does the existing heatmap carry sub-cell position
information around its peaks? No training, one val pass with the start
weights.

For every initial query (K = num_proposals) the 3x3 neighbourhood of its
class channel (sigmoid) gives three positions:
  cell   the cell centre (what TransFusion uses)
  dark   DARK: Newton step on log(heat), finite differences, each component
         clipped to +-0.5 cell; falls back to the cell centre on the map
         border and where the Hessian is not negative definite
  soft   3x3 soft-argmax with the sigmoid values as weights
Every GT (class-aware evaluation ranges, as D1) is attached to its nearest
same-class query by cell-centre distance (within 2 m) and grouped by that
distance: G0 <= 0.5 m, G1 (0.5, 1] m, G2 (1, 2] m; bins <30 / >=30 m.

Outputs (single report pair, plus per-GT arrays):
  <out-dir>/subcell_0a.txt / .json   tables of exec doc #3 sec. 5.1,
                                     step 0b trigger, self-checks S-0a1/2
  <out-dir>/subcell_0a_per_gt.npz
Self-checks: S-0a1 cell-centre init recall @0.5 m (all GT) equals the B0
value of exec doc #2 (same weights, same definition) within --b0-tol;
S-0a2 replicated top-K labels == head.query_labels for >= 99%.

Usage (from the workspace root; run_queryloc.sh 0a wraps it):
  python projects/QueryLoc/tools/subcell_0a.py CONFIG CKPT \
      --out-dir work_dirs/queryloc/step0a [--cfg-options ...]
"""
import argparse
import importlib.util
import json
import os
import os.path as osp
import sys
import time

import numpy as np
import torch

HERE = osp.dirname(osp.abspath(__file__))
QL_ROOT = osp.dirname(HERE)
QD_DEFAULT = osp.join(osp.dirname(osp.dirname(QL_ROOT)), 'tools',
                      'query_diag')
NO_NMS_CLASSES = (8, 9)          # nuScenes pedestrian, traffic_cone
POS = ('cell', 'dark', 'soft')
GROUPS = (('G0', 0.0, 0.5), ('G1', 0.5, 1.0), ('G2', 1.0, 2.0))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_args():
    from mmengine.config import DictAction
    p = argparse.ArgumentParser(description='exec doc #3 step 0a')
    p.add_argument('config')
    p.add_argument('checkpoint')
    p.add_argument('--out-dir', default='work_dirs/queryloc/step0a')
    p.add_argument('--qd', default=QD_DEFAULT,
                   help='tools/query_diag (diag_common.py is reused)')
    p.add_argument('--max-samples', type=int, default=-1)
    p.add_argument('--attach-thr', type=float, default=2.0,
                   help='max distance (m) GT -> nearest same-class query')
    p.add_argument('--recall-thr', type=float, default=0.5)
    p.add_argument('--near-far', type=float, default=30.0)
    p.add_argument('--clamp', type=float, default=0.5,
                   help='DARK clip per component (cells)')
    p.add_argument('--trigger-pp', type=float, default=0.5,
                   help='step 0b runs if DARK raises recall by >= this (pp)')
    p.add_argument('--b0-init05', type=float, default=0.82792,
                   help='B0 init recall @0.5 m (all GT), exec doc #2')
    p.add_argument('--b0-tol', type=float, default=0.001)
    p.add_argument('--no-class-range', action='store_true')
    p.add_argument('--cfg-options', nargs='+', action=DictAction)
    return p.parse_args()


def stats(a):
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return {'n': 0, 'mean': float('nan'), 'median': float('nan')}
    return {'n': int(a.size), 'mean': float(a.mean()),
            'median': float(np.median(a))}


def summarize(R, Q, args, P):
    dist, grp = R['dist'], R['group']
    bins = {'<%g' % args.near_far: dist < args.near_far,
            '>=%g' % args.near_far: dist >= args.near_far,
            'all': np.ones_like(dist, dtype=bool)}
    out = {'groups': {}, 'recall': {}, 'queries': {}}
    for g, _, _ in GROUPS:
        out['groups'][g] = {}
        for bname, bm in bins.items():
            m = bm & (grp == g)
            row = {'n': int(m.sum())}
            for k in POS:
                row['d_' + k] = stats(R['d_' + k][m])
            row['moved_dark'] = float(np.mean(R['d_dark'][m] <
                                              R['d_cell'][m])) \
                if m.any() else float('nan')
            row['moved_soft'] = float(np.mean(R['d_soft'][m] <
                                              R['d_cell'][m])) \
                if m.any() else float('nan')
            st = R['status'][m]
            row['fallback'] = {'border': int((st == 1).sum()),
                               'not_neg_def': int((st == 2).sum())}
            row['dark_clipped'] = int(R['clipped'][m].sum())
            out['groups'][g][bname] = row
    for bname, bm in bins.items():
        r = {k: float(np.mean(R['hit_' + k][bm])) if bm.any() else
             float('nan') for k in POS}
        r['n'] = int(bm.sum())
        r['d_dark_minus_cell_pp'] = 100 * (r['dark'] - r['cell'])
        r['d_soft_minus_cell_pp'] = 100 * (r['soft'] - r['cell'])
        out['recall'][bname] = r
    for name in ('all', 'nms', 'no_nms'):
        q = Q[name]
        out['queries'][name] = dict(q)
    d_all = out['recall']['all']['d_dark_minus_cell_pp']
    out['trigger_0b'] = {'d_dark_minus_cell_pp': d_all,
                         'threshold_pp': args.trigger_pp,
                         'triggered': bool(d_all >= args.trigger_pp)}
    return out


def report_lines(S, args, P, checks):
    L = ['Exec doc #3 step 0a: sub-cell position in the existing heatmap '
         '(K=%d, class-aware, recall threshold %.2g m)' % (P, args.recall_thr)]
    L.append('')
    L.append('Initial recall @%.2g m by position type (all queries)'
             % args.recall_thr)
    L.append('%-6s %7s %8s %8s %8s %12s %12s'
             % ('bin', 'N_gt', 'cell', 'dark', 'soft', 'dark-cell pp',
                'soft-cell pp'))
    for b, r in S['recall'].items():
        L.append('%-6s %7d %8.5f %8.5f %8.5f %+12.3f %+12.3f'
                 % (b, r['n'], r['cell'], r['dark'], r['soft'],
                    r['d_dark_minus_cell_pp'], r['d_soft_minus_cell_pp']))
    L.append('')
    L.append('GT attached to the nearest same-class query (cell centre, '
             '<= %.3g m); distances in m, mean / median' % args.attach_thr)
    L.append('%-3s %-6s %7s %17s %17s %17s %7s %7s %7s %7s %7s'
             % ('grp', 'bin', 'N', 'd_cell', 'd_dark', 'd_soft',
                'moved_d', 'moved_s', 'fb_bdr', 'fb_nnd', 'clip'))
    for g, rows in S['groups'].items():
        for b, r in rows.items():
            L.append('%-3s %-6s %7d %8.4f/%8.4f %8.4f/%8.4f %8.4f/%8.4f '
                     '%7.4f %7.4f %7d %7d %7d'
                     % (g, b, r['n'], r['d_cell']['mean'],
                        r['d_cell']['median'], r['d_dark']['mean'],
                        r['d_dark']['median'], r['d_soft']['mean'],
                        r['d_soft']['median'], r['moved_dark'],
                        r['moved_soft'], r['fallback']['border'],
                        r['fallback']['not_neg_def'], r['dark_clipped']))
    L.append('  moved_d / moved_s: share with d_dark / d_soft < d_cell; '
             'fb_bdr / fb_nnd: DARK fell back (map border / Hessian not '
             'negative definite); clip: DARK offset clipped to +-%.2g cell'
             % args.clamp)
    L.append('')
    L.append('DARK status over all selected queries')
    for name, q in S['queries'].items():
        L.append('  %-7s queries %9d  ok %9d  border %7d  not_neg_def %8d  '
                 'clipped %8d' % (name, q['n'], q['ok'], q['border'],
                                  q['not_neg_def'], q['clipped']))
    L.append('  (nms = classes with 3x3 local max; no_nms = pedestrian, '
             'traffic_cone)')
    L.append('')
    t = S['trigger_0b']
    L.append('Step 0b trigger: dark - cell recall @%.2g m (all) = %+.3f pp '
             '(threshold %+.2f pp) -> %s'
             % (args.recall_thr, t['d_dark_minus_cell_pp'], t['threshold_pp'],
                'TRIGGERED' if t['triggered'] else 'not triggered'))
    L.append('')
    L.append('Self-checks')
    for c in checks:
        L.append('  %-6s %-52s %-5s %s' % (c['id'], c['desc'], c['result'],
                                           json.dumps(c['value'])))
    return L


def main():
    t_start = time.time()
    args = parse_args()
    dc = load_module('diag_common', osp.join(args.qd, 'diag_common.py'))
    sc = load_module('ql_subcell', osp.join(QL_ROOT, 'queryloc',
                                            'subcell.py'))
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmengine.runner.amp import autocast

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
        print('[0a] NOTE: --cfg-options in effect (recorded in json):',
              args.cfg_options)
    fp16 = dc.loop_fp16(cfg, 'test_cfg')
    cfg.work_dir = args.out_dir
    cfg.launcher = 'none'
    cfg.load_from = None
    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(args.checkpoint, map_location='cpu')
    model = runner.model
    model.cuda().eval()
    dl = runner.build_dataloader(cfg.test_dataloader)
    class_names = dc.get_class_names(cfg, dl.dataset)
    head = dc.install_capture(model)
    P = int(head.num_proposals)
    osf, vx, vy, _, _ = dc.head_geometry(head)
    print('[0a] num_proposals=%d test_cfg.fp16=%s geometry=%s'
          % (P, fp16, dc.head_geometry(head)))

    keys = ('dist', 'label', 'frame', 'group', 'd_cell', 'd_dark', 'd_soft',
            'status', 'clipped', 'hit_cell', 'hit_dark', 'hit_soft')
    R = {k: [] for k in keys}
    Q = {n: {'n': 0, 'ok': 0, 'border': 0, 'not_neg_def': 0, 'clipped': 0}
         for n in ('all', 'nms', 'no_nms')}
    match_hit = match_tot = 0
    transposed = None
    n_frames = 0
    t0 = time.time()
    last_print = 0
    with torch.no_grad():
        for data in dl:
            if 0 < args.max_samples <= n_frames:
                break
            with autocast(enabled=fp16):
                model.test_step(data)
            S = dc.STATE
            logits = S['logits']
            B, C, H, W = logits.shape
            if transposed is None:
                transposed = dc.probe_bev_pos(head, H, W)
                print('[0a] BEV map %dx%d, bev_pos transposed=%s'
                      % (H, W, transposed))
            heat = logits.detach().float().sigmoid()
            idx, cls, _ = dc.replicate_init(head, logits, P)
            ql = S['query_labels']
            match_hit += int((cls == ql).sum().item())
            match_tot += int(ql.numel())
            nb, valid, border = sc.neighborhood3x3(heat, idx, cls)
            d_off, status, clipped = sc.dark_offsets(nb, border,
                                                     clamp=args.clamp)
            s_off = sc.softargmax_offsets(nb, valid)
            scale = torch.tensor([osf * vx, osf * vy])
            q_cell = dc.cells_to_xy(head, idx).cpu()            # [B, P, 2]
            q_pos = {'cell': q_cell,
                     'dark': q_cell + sc.rowcol_to_xy(
                         d_off.cpu(), transposed) * scale,
                     'soft': q_cell + sc.rowcol_to_xy(
                         s_off.cpu(), transposed) * scale}
            st, cl, cc = status.cpu(), clipped.cpu(), cls.cpu()
            for name, m in (('all', torch.ones_like(cc, dtype=torch.bool)),
                            ('no_nms', (cc == 8) | (cc == 9))):
                Q[name]['n'] += int(m.sum())
                Q[name]['ok'] += int((m & (st == sc.OK)).sum())
                Q[name]['border'] += int((m & (st == sc.BORDER)).sum())
                Q[name]['not_neg_def'] += int(
                    (m & (st == sc.NOT_NEG_DEF)).sum())
                Q[name]['clipped'] += int((m & cl).sum())
            for b, sample in enumerate(S['samples']):
                fid = n_frames
                n_frames += 1
                g_xy, g_lab, g_dist = dc.gt_from_sample(
                    sample, class_names, not args.no_class_range)
                if len(g_lab) == 0:
                    continue
                hits = {}
                for k in POS:
                    _, rc = dc.per_gt_hits(g_xy, g_lab, q_pos[k][b], cc[b],
                                           [args.recall_thr])
                    hits[k] = (rc[:, 0] < P).numpy()
                D = torch.cdist(g_xy.double(), q_cell[b].double())
                same = cc[b][None, :] == g_lab[:, None]
                D = torch.where(same, D, torch.full_like(D, float('inf')))
                dmin, j = D.min(1)
                grp = np.full(len(g_lab), '', dtype='<U2')
                d = dmin.numpy()
                for gname, lo, hi in GROUPS:
                    grp[(d > lo if lo > 0 else d >= 0) & (d <= hi)] = gname
                grp[d > args.attach_thr] = ''
                for k in POS:
                    R['d_' + k].append(torch.linalg.norm(
                        g_xy.double() - q_pos[k][b][j].double(),
                        dim=1).numpy())
                R['dist'].append(g_dist.numpy())
                R['label'].append(g_lab.numpy())
                R['frame'].append(np.full(len(g_lab), fid))
                R['group'].append(grp)
                R['status'].append(st[b][j].numpy())
                R['clipped'].append(cl[b][j].numpy())
                for k in POS:
                    R['hit_' + k].append(hits[k])
            if n_frames // 500 != last_print:
                last_print = n_frames // 500
                print('[0a] %d frames, %.1fs' % (n_frames, time.time() - t0))
    Q['nms'] = {k: Q['all'][k] - Q['no_nms'][k] for k in Q['all']}
    R = {k: np.concatenate(v) if v else np.zeros(0) for k, v in R.items()}
    S = summarize(R, Q, args, P)
    cell_all = S['recall']['all']['cell']
    match = match_hit / max(match_tot, 1)
    checks = [
        {'id': 'S-0a1', 'desc': 'cell recall @%.2g m (all) == B0 %.5f '
         '(tol %.3g)' % (args.recall_thr, args.b0_init05, args.b0_tol),
         'value': {'cell': cell_all, 'b0': args.b0_init05,
                   'diff': cell_all - args.b0_init05},
         'result': 'PASS' if abs(cell_all - args.b0_init05) <= args.b0_tol
         else 'FAIL'},
        {'id': 'S-0a2', 'desc': 'replicated top-K labels == '
         'head.query_labels (>= 0.99)', 'value': match,
         'result': 'PASS' if match >= 0.99 else 'FAIL'}]
    L = report_lines(S, args, P, checks)
    ql_md5 = {osp.relpath(p, QL_ROOT): dc.file_md5(p) for p in (
        osp.join(QL_ROOT, 'queryloc', 'subcell.py'), osp.abspath(__file__))}
    res = {'args': vars(args), 'num_proposals': P, 'frames': n_frames,
           'transposed': transposed, 'summary': S, 'self_checks': checks,
           'runtime': dc.runtime_info(cfg, args.config, args.checkpoint,
                                      t_start, args.cfg_options,
                                      extra={'queryloc_md5': ql_md5})}
    text = '\n'.join(L)
    print(text)
    with open(osp.join(args.out_dir, 'subcell_0a.txt'), 'w') as f:
        f.write(text + '\n')
    with open(osp.join(args.out_dir, 'subcell_0a.json'), 'w') as f:
        json.dump(res, f, indent=2, default=str)
    np.savez_compressed(osp.join(args.out_dir, 'subcell_0a_per_gt.npz'),
                        **R)
    print('[0a] saved %s/subcell_0a.{txt,json} and subcell_0a_per_gt.npz '
          '(%d frames, %.1fs)' % (args.out_dir, n_frames,
                                  time.time() - t_start))
    return 0 if all(c['result'] == 'PASS' for c in checks) else 1


if __name__ == '__main__':
    sys.exit(main())
