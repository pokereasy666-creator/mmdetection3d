#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Diagnostics 1 + 3 for the TransFusion head of BEVFusion (mmdet3d 1.4).

1. Initial-query recall: for every nuScenes val GT, is there an initial
   query (heatmap local-max NMS + global top-K) within t meters of its
   center? Reported class-agnostic and class-aware, by ego distance and
   class, for several K (to see whether a larger / re-allocated query
   budget could recover misses), and also after the decoder layer.
3. Per-frame GT count distribution vs. num_proposals.

No training, no source modification. Single GPU.

Example (run from the mmdetection3d repo root):
  PYTHONPATH=$PWD:/path/to/diag python /path/to/diag/diag_query_recall.py \
      projects/BEVFusion/configs/<your_config>.py /path/to/ckpt.pth \
      --out-dir work_dirs/diag/recall --max-samples 300
"""
import argparse
import json
import os
import os.path as osp
import sys
import time

import numpy as np
import torch

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import diag_common as dc  # noqa: E402


def parse_args():
    from mmengine.config import DictAction
    p = argparse.ArgumentParser(description='TransFusion query recall')
    p.add_argument('config')
    p.add_argument('checkpoint')
    p.add_argument('--out-dir', default='work_dirs/diag/recall')
    p.add_argument('--max-samples', type=int, default=-1,
                   help='stop after N frames (-1 = full val set)')
    p.add_argument('--ks', default='200,300,500,1000',
                   help='query budgets to evaluate (num_proposals added)')
    p.add_argument('--thr', default='0.5,1,2,4',
                   help='center-distance thresholds in meters')
    p.add_argument('--bins', default='0,20,30,40,50',
                   help='ego-distance bin edges in meters')
    p.add_argument('--main-thr', type=float, default=2.0,
                   help='threshold used for the per-class / per-K tables')
    p.add_argument('--no-class-range', action='store_true',
                   help='do not apply nuScenes per-class eval ranges')
    p.add_argument('--cfg-options', nargs='+', action=DictAction)
    return p.parse_args()


# ---------------------------------------------------------------------------
# aggregation / printing
# ---------------------------------------------------------------------------

def bin_masks(dist, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        out.append(('%g-%g' % (lo, hi), (dist >= lo) & (dist < hi)))
    out.append(('all', np.ones_like(dist, dtype=bool)))
    return out


def rate(x):
    return float(np.mean(x)) if len(x) else float('nan')


def fmt(v):
    return '  n/a ' if v != v else '%6.3f' % v


def summarize(R, frame_counts, class_names, ks, thr, edges, P, main_thr):
    dist, lab = R['dist'], R['label']
    ra, rc = R['rank_agn'], R['rank_ca']          # [N, T]
    fa, fc = R['fin_agn'], R['fin_ca']            # [N, T] bool
    ti = thr.index(main_thr) if main_thr in thr else len(thr) // 2
    bins = bin_masks(dist, edges)
    S = {'num_gt': int(len(dist)), 'num_proposals': P, 'thr': thr,
         'main_thr': thr[ti], 'ks': ks, 'bins': [b[0] for b in bins]}
    lines = []

    def emit(s=''):
        lines.append(s)

    # T1: init recall at K=P vs threshold
    for tag, rk in (('class-aware', rc), ('class-agnostic', ra)):
        emit('T1 [%s] initial-query recall @K=%d (num_proposals)' % (tag, P))
        emit('%-8s %7s ' % ('bin', 'N_gt') +
             ' '.join('%6sm' % ('%g' % t) for t in thr))
        tab = {}
        for name, m in bins:
            row = [rate(rk[m, j] < P) for j in range(len(thr))]
            tab[name] = {'n': int(m.sum()), 'recall': row}
            emit('%-8s %7d ' % (name, m.sum()) +
                 ' '.join(fmt(v) for v in row))
        S['T1_' + tag] = tab
        emit()

    # T2: per class x bin, class-aware @main_thr, K=P
    emit('T2 class-aware initial recall @%gm, K=%d, per class (N in [])'
         % (thr[ti], P))
    emit('%-22s' % 'class' + ''.join('%15s' % b[0] for b in bins))
    tab = {}
    for ci, cn in enumerate(class_names):
        cells, rowd = [], {}
        for name, m in bins:
            mm = m & (lab == ci)
            v = rate(rc[mm, ti] < P)
            rowd[name] = {'n': int(mm.sum()), 'recall': v}
            cells.append('%8s[%5d]' % (fmt(v).strip(), mm.sum()))
        tab[cn] = rowd
        emit('%-22s' % cn + ''.join('%15s' % x for x in cells))
    S['T2'] = tab
    emit()

    # T3: recall vs K
    for tag, rk in (('class-aware', rc), ('class-agnostic', ra)):
        emit('T3 [%s] initial recall @%gm vs query budget K' % (tag, thr[ti]))
        emit('%-8s ' % 'bin' + ' '.join('%7s' % ('K=%d' % k) for k in ks))
        tab = {}
        for name, m in bins:
            row = [rate(rk[m, ti] < k) for k in ks]
            tab[name] = row
            emit('%-8s ' % name + ' '.join(' %6.3f' % v for v in row))
        S['T3_' + tag] = tab
        emit()

    # T4: after decoder vs init
    emit('T4 recall @%gm: initial query vs. after decoder (final centers)'
         % thr[ti])
    emit('%-8s %10s %10s %10s %10s' % ('bin', 'init_ca', 'final_ca',
                                       'init_agn', 'final_agn'))
    tab = {}
    for name, m in bins:
        row = [rate(rc[m, ti] < P), rate(fc[m, ti]),
               rate(ra[m, ti] < P), rate(fa[m, ti])]
        tab[name] = row
        emit('%-8s ' % name + ' '.join('%10.3f' % v for v in row))
    S['T4'] = tab
    emit()

    # T5: per-frame GT counts
    fcnt = np.asarray(frame_counts, dtype=np.int64)
    if len(fcnt):
        q = np.percentile(fcnt, [50, 95, 99])
        S['T5'] = {'frames': int(len(fcnt)), 'mean': float(fcnt.mean()),
                   'p50': float(q[0]), 'p95': float(q[1]),
                   'p99': float(q[2]), 'max': int(fcnt.max()),
                   'frac_gt_over_P': float((fcnt > P).mean()),
                   'frac_gt_over_P_half': float((fcnt > P // 2).mean())}
        emit('T5 GT per frame (valid, inside BEV grid): frames=%d mean=%.1f '
             'p50=%g p95=%g p99=%g max=%d | frac>P=%.4f frac>P/2=%.4f'
             % (len(fcnt), fcnt.mean(), q[0], q[1], q[2], fcnt.max(),
                (fcnt > P).mean(), (fcnt > P // 2).mean()))
    return S, lines


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    args = parse_args()
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmengine.runner.amp import autocast

    thr = [float(x) for x in args.thr.split(',')]
    edges = [float(x) for x in args.bins.split(',')]
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
        print('[diag] NOTE: --cfg-options in effect (recorded in json):',
              args.cfg_options)
    # Follow the official TestLoop: autocast only if test_cfg.fp16 is set.
    fp16 = dc.loop_fp16(cfg, 'test_cfg')
    print('[diag] test_cfg.fp16=%s (recall loop autocast=%s)' % (fp16, fp16))
    cfg.work_dir = args.out_dir
    cfg.launcher = 'none'
    cfg.load_from = None
    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(args.checkpoint, map_location='cpu')
    model = runner.model
    model.cuda().eval()
    dl = runner.build_dataloader(cfg.test_dataloader)
    class_names = dc.get_class_names(cfg, dl.dataset)
    print('[diag] classes:', class_names)

    head = dc.install_capture(model)
    P = int(head.num_proposals)
    ks = sorted(set([P] + [int(k) for k in args.ks.split(',')]))
    kmax = max(ks)
    print('[diag] num_proposals=%d nms_kernel=%d dataset=%s geometry=%s'
          % (P, head.nms_kernel_size, head.test_cfg.get('dataset'),
             dc.head_geometry(head)))

    R = {k: [] for k in ('dist', 'label', 'rank_agn', 'rank_ca',
                         'fin_agn', 'fin_ca', 'frame')}
    frame_counts = []
    match_hit = match_tot = 0
    cg_in = cg_tot = 0          # S4a: center_grid values in grid units
    transposed = None
    n_frames = 0
    last_print = 0
    t0 = time.time()
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
                print('[diag] BEV map %dx%d, bev_pos transposed=%s'
                      % (H, W, transposed))
            idx, cls, _ = dc.replicate_init(head, logits, kmax)
            ql = S['query_labels']
            match_hit += int((cls[:, :P] == ql).sum().item())
            match_tot += int(ql.numel())
            q_xy = dc.cells_to_xy(head, idx).cpu()          # [B, kmax, 2]
            # Decoder centers from the pre-decode snapshot (grid units);
            # head_out['center'] is metric by now (decode works in place).
            cg = S['center_grid'].float()                   # [B, 2, P]
            hi = cg.new_tensor([H + 2.0, W + 2.0]).view(1, 2, 1)
            cg_in += int(((cg >= -2.0) & (cg <= hi)).sum().item())
            cg_tot += int(cg.numel())
            fin_xy = dc.grid_to_xy(head, cg).cpu()
            for b, sample in enumerate(S['samples']):
                fid = n_frames
                n_frames += 1
                g_all, _, _ = dc.gt_from_sample(sample, class_names, False)
                _, _, v = dc.xy_to_flat(head, g_all, H, W, transposed)
                frame_counts.append(int(v.sum()))
                g_xy, g_lab, g_dist = dc.gt_from_sample(
                    sample, class_names, not args.no_class_range)
                if len(g_lab) == 0:
                    continue
                ra, rc = dc.per_gt_hits(g_xy, g_lab, q_xy[b],
                                        cls[b].cpu(), thr)
                fra, frc = dc.per_gt_hits(g_xy, g_lab, fin_xy[b],
                                          ql[b].cpu(), thr)
                R['dist'].append(g_dist.numpy())
                R['label'].append(g_lab.numpy())
                R['rank_agn'].append(ra.numpy())
                R['rank_ca'].append(rc.numpy())
                R['fin_agn'].append((fra < P).numpy())
                R['fin_ca'].append((frc < P).numpy())
                R['frame'].append(np.full(len(g_lab), fid))
            if n_frames // 200 != last_print:
                last_print = n_frames // 200
                print('[diag] %d frames, %.1fs' % (n_frames,
                                                   time.time() - t0))

    R = {k: np.concatenate(v) if len(v) else np.zeros(0)
         for k, v in R.items()}
    match = match_hit / max(match_tot, 1)
    cg_frac = cg_in / max(cg_tot, 1)
    summ, lines = summarize(R, frame_counts, class_names, ks, thr, edges,
                            P, args.main_thr)
    summ['init_label_match_rate'] = match
    summ['frames'] = n_frames
    summ['bev_pos_transposed'] = transposed
    summ['class_range'] = not args.no_class_range
    summ['center_grid_in_range_frac'] = cg_frac
    summ['center_grid_values'] = cg_tot

    head_lines = ['[sanity] replicated init vs head.query_labels match rate: '
                  '%.5f' % match,
                  '[sanity] center_grid values within [-2, H+2] grid units: '
                  '%.5f (of %d)' % (cg_frac, cg_tot)]
    if match < 0.99:
        head_lines.append('[WARNING] match < 0.99: your transfusion_head.py '
                          'query init differs from the replicated rule; the '
                          'recall numbers are NOT trustworthy.')
    near = summ['T1_class-agnostic'].get(summ['bins'][0], {})
    if near and near['recall'][-1] < 0.5:
        head_lines.append('[WARNING] near-range class-agnostic recall @%gm is '
                          '%.3f (<0.5): check the BEV axis convention.'
                          % (thr[-1], near['recall'][-1]))
    text = '\n'.join(head_lines + [''] + lines)
    print(text)
    summ['runtime'] = dc.runtime_info(
        cfg, args.config, args.checkpoint, t_start,
        cfg_options=args.cfg_options,
        extra={'recall_loop_fp16': fp16, 'frames': n_frames,
               'loop_s': round(time.time() - t0, 1)})
    rt = summ['runtime']
    print('[diag] runtime: %.1fs total, %.1fs loop, %d frames, peak mem '
          'allocated %s MiB / reserved %s MiB'
          % (rt['elapsed_s'], rt['loop_s'], n_frames,
             rt.get('peak_mem_allocated_mib', 'n/a'),
             rt.get('peak_mem_reserved_mib', 'n/a')))
    with open(osp.join(args.out_dir, 'recall_report.txt'), 'w') as f:
        f.write(text + '\n')
    with open(osp.join(args.out_dir, 'recall_summary.json'), 'w') as f:
        json.dump(summ, f, indent=2, default=str)
    np.savez_compressed(osp.join(args.out_dir, 'recall_per_gt.npz'),
                        frame_counts=np.asarray(frame_counts),
                        thr=np.asarray(thr), ks=np.asarray(ks), P=P,
                        class_names=np.asarray(class_names),
                        class_range=not args.no_class_range, **R)
    print('[diag] saved to', args.out_dir)


if __name__ == '__main__':
    main()
