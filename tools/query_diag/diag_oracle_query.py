#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Diagnostic 2: oracle query-initialization upper bound for BEVFusion.

At test time, GT centers (from data_sample.eval_ann_info) are injected into
the TransFusion head's query initialization by boosting the dense heatmap
logit at the GT cell (forward hook on bbox_head.heatmap_head). The head's
own local-max NMS + top-K then selects those cells first; the remaining
slots are filled by the normal peaks, so the query budget stays the same.

To avoid leaking GT confidence, the heatmap score of injected queries
(query_heatmap_score, multiplied into the final score) is restored to the
ORIGINAL pre-NMS sigmoid value of that cell (--score orig, default).

Variants (see run_diag.sh):
  --oracle none                      baseline (must reproduce your mAP/NDS)
  --oracle loc     --range 0 1000    location oracle, all GT
  --oracle loc     --range 30 1000   location oracle, far GT only
  --oracle loc     --range 0 30      location oracle, near GT only
  --oracle loc_cls --range 0 1000    location + class oracle
Exec doc #2 adds:
  --subset missed [--miss-thr 2.0]   inject only GT that have no same-class
                                     initial query within miss-thr meters
                                     on the UNMODIFIED heatmap (K =
                                     num_proposals; same functions as D1)
  --score boost / --jitter-cells J   (already supported)

The mAP/NDS gain of a variant over 'none' is the ceiling for ANY method
that only changes query allocation / initialization (Dome-DETR PAQI, ALIGN,
FocalFormer3D HIP, ...).

Supports single GPU or torchrun (--launcher pytorch). Uses the config's
test_evaluator, i.e. the official nuScenes metric.
"""
import argparse
import json
import os
import os.path as osp
import sys
import time

import torch

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import diag_common as dc  # noqa: E402

BOOST_LOGIT = 30.0   # sigmoid(30) == 1.0 in fp32: always selected first


def parse_args():
    from mmengine.config import DictAction
    p = argparse.ArgumentParser(description='oracle query init upper bound')
    p.add_argument('config')
    p.add_argument('checkpoint')
    p.add_argument('--work-dir', default='work_dirs/diag/oracle')
    p.add_argument('--tag', default='run')
    p.add_argument('--oracle', choices=['none', 'loc', 'loc_cls'],
                   default='loc')
    p.add_argument('--range', nargs=2, type=float, default=[0.0, 1000.0],
                   metavar=('MIN', 'MAX'),
                   help='inject only GT with ego distance in [MIN, MAX) m')
    p.add_argument('--score', choices=['orig', 'boost'], default='orig',
                   help='orig: restore original heatmap score of injected '
                        'queries (no confidence leak); boost: keep ~1.0')
    p.add_argument('--subset', choices=['all', 'missed'], default='all',
                   help='all: every GT in range; missed: only GT without a '
                        'same-class initial query within --miss-thr m')
    p.add_argument('--miss-thr', type=float, default=2.0,
                   help='center distance (m) defining a missed GT')
    p.add_argument('--jitter-cells', type=int, default=0,
                   help='random +-J cell offset on injected positions')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no-class-range', action='store_true')
    p.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
                   default='none')
    p.add_argument('--local_rank', '--local-rank', type=int, default=0)
    p.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = p.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    t_start = time.time()
    args = parse_args()
    from mmengine.config import Config
    from mmengine.dist import all_reduce, get_rank
    from mmengine.runner import Runner

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
        print('[oracle] NOTE: --cfg-options in effect (recorded in json):',
              args.cfg_options)
    cfg.launcher = args.launcher
    cfg.work_dir = args.work_dir
    cfg.load_from = args.checkpoint
    runner = Runner.from_cfg(cfg)
    model = dc.unwrap(runner.model)
    head = dc.install_capture(model)
    class_names = dc.get_class_names(cfg)
    P = int(head.num_proposals)
    rmin, rmax = args.range
    gen = torch.Generator().manual_seed(args.seed + get_rank())

    # counters: 0 frames, 1 injected cells, 2 capped, 3 restored,
    # 4 init-mismatch, 5 P*B, 6 selected_injected, 7 gt_in_range,
    # 8 missed_gt
    cnt = torch.zeros(9, dtype=torch.float64)
    cache = {}

    def logits_modifier(logits):
        if args.oracle == 'none':
            return logits
        S = dc.STATE
        B, C, H, W = logits.shape
        if 'transposed' not in cache:
            cache['transposed'] = dc.probe_bev_pos(head, H, W)
        out = logits.clone()
        mask = torch.zeros(B, C * H * W, dtype=torch.bool,
                           device=logits.device)
        # Initial queries of the UNMODIFIED heatmap (as in D1), used to
        # define missed GT: no same-class query within miss-thr meters.
        idx0, cls0, _ = dc.replicate_init(head, logits, P)
        q_xy0 = dc.cells_to_xy(head, idx0).cpu()
        cls0 = cls0.cpu()
        for b, sample in enumerate(S['samples']):
            xy, lab, dist = dc.gt_from_sample(sample, class_names,
                                              not args.no_class_range)
            keep = (dist >= rmin) & (dist < rmax)
            xy, lab = xy[keep], lab[keep]
            missed = torch.zeros(len(lab), dtype=torch.bool)
            if len(lab):
                _, rank_ca = dc.per_gt_hits(xy, lab, q_xy0[b], cls0[b],
                                            [args.miss_thr])
                missed = rank_ca[:, 0] >= P
            cnt[7] += len(lab)
            cnt[8] += int(missed.sum())
            if args.subset == 'missed':
                xy, lab = xy[missed], lab[missed]
            ch, r, c, n_cap = dc.oracle_cells(
                head, logits[b], xy, lab, H, W,
                cache['transposed'], args.oracle, P,
                jitter=args.jitter_cells, gen=gen)
            cnt[0] += 1
            cnt[1] += len(ch)
            cnt[2] += n_cap
            if len(ch) == 0:
                continue
            dev = logits.device
            ch, r, c = ch.to(dev), r.to(dev), c.to(dev)
            out[b][ch, r, c] = BOOST_LOGIT
            mask[b, ch * (H * W) + r * W + c] = True
        S['oracle_mask'] = mask
        return out

    def head_out_modifier(head_out):
        if args.oracle == 'none':
            return
        S = dc.STATE
        d = dc.first_dict(head_out)
        logits = S['logits']
        idx, cls, _ = dc.replicate_init(head, logits, P)
        ql = head.query_labels
        cnt[4] += float((cls != ql).sum().item())
        cnt[5] += float(ql.numel())
        B, C, H, W = logits.shape
        # selected initial queries whose (class, cell) is an injected one
        sel = S['oracle_mask'].gather(1, cls * (H * W) + idx)
        cnt[6] += float(sel.sum().item())
        if args.score == 'orig':
            sig_raw = S['logits_raw'].float().sigmoid().view(B, C, -1)
            cnt[3] += dc.restore_oracle_scores(
                d['query_heatmap_score'], cls, idx, S['oracle_mask'], sig_raw)

    dc.HOOKS['logits_modifier'] = logits_modifier
    dc.HOOKS['head_out_modifier'] = head_out_modifier

    metrics = runner.test()

    t = cnt.cuda() if torch.cuda.is_available() else cnt
    all_reduce(t)
    t = t.cpu().tolist()
    stats = {'frames': t[0], 'injected_cells': t[1], 'capped_cells': t[2],
             'restored_queries': t[3],
             'init_mismatch_rate': t[4] / max(t[5], 1.0),
             'selected_injected': t[6], 'gt_in_range': t[7],
             'missed_gt': t[8]}
    if get_rank() == 0:
        res = {'args': vars(args), 'num_proposals': P, 'stats': stats,
               'metrics': {k: (float(v) if isinstance(v, (int, float))
                               else str(v)) for k, v in metrics.items()}}
        res['runtime'] = dc.runtime_info(cfg, args.config, args.checkpoint,
                                         t_start,
                                         cfg_options=args.cfg_options)
        os.makedirs(args.work_dir, exist_ok=True)
        path = osp.join(args.work_dir, 'oracle_%s.json' % args.tag)
        with open(path, 'w') as f:
            json.dump(res, f, indent=2, default=str)
        print('[oracle] tag=%s mode=%s subset=%s miss_thr=%g range=[%g,%g) '
              'score=%s jitter=%d seed=%d'
              % (args.tag, args.oracle, args.subset, args.miss_thr, rmin,
                 rmax, args.score, args.jitter_cells, args.seed))
        print('[oracle] stats:', json.dumps(stats))
        rt = res['runtime']
        print('[oracle] runtime: %.1fs, peak mem allocated %s MiB / '
              'reserved %s MiB, test_cfg.fp16=%s'
              % (rt['elapsed_s'], rt.get('peak_mem_allocated_mib', 'n/a'),
                 rt.get('peak_mem_reserved_mib', 'n/a'),
                 rt['fp16']['test_cfg']))
        if args.oracle != 'none' and stats['init_mismatch_rate'] > 0.01:
            print('[WARNING] replicated init differs from head.query_labels '
                  '(rate %.4f); score restoration may be wrong.'
                  % stats['init_mismatch_rate'])
        if stats['frames'] and stats['selected_injected'] < \
                0.99 * stats['injected_cells']:
            print('[WARNING] selected_injected < 0.99 x injected_cells; some '
                  'injected cells did not become initial queries.')
        if stats['frames'] and stats['restored_queries'] < \
                0.99 * stats['injected_cells'] and args.score == 'orig':
            print('[WARNING] fewer restored queries than injected cells; '
                  'some injected cells were not selected (check top-K).')
        for k, v in sorted(res['metrics'].items()):
            if k.endswith('mAP') or k.endswith('NDS'):
                print('[oracle] %s = %s' % (k, v))
        print('[oracle] saved', path)


if __name__ == '__main__':
    main()
