#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exec doc #3: single-GPU val evaluation of a QueryLoc config, written in
the oracle json format so that tools/query_diag/oracle_breakdown.py
(unchanged) compares it with the exec doc #2 E0 json.

  S-M0  ql_mb.py on the start weights (zero offsets): --debug-frames 50 runs
        the bitwise check first (query features of the bilinear path ==
        the original gather, positions == cell centres, every frame), then
        the full val; --tol 0.001 checks |dmAP|, |dNDS| vs E0.
  0b    ql_0b.py on the start weights (DARK positions, peak-cell features):
        dmAP / dNDS and the per-threshold breakdown vs E0 (no tolerance).
Usage: python projects/QueryLoc/tools/eval_ql.py CONFIG CKPT --tag TAG \
           --out-dir DIR --ref-json E0.json [--tol 0.001] \
           [--debug-frames 50] [--cfg-options ...]
"""
import argparse
import json
import os
import os.path as osp
import sys
import time

import torch

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import ql_common as qc  # noqa: E402


def debug_frames(runner, cfg, n):
    """S-M0 bitwise check on the first n val frames."""
    model = runner.model
    model.cuda().eval()
    head = qc.unwrap(model).bbox_head
    head.debug_check_gather = True
    dl = runner.build_dataloader(cfg.test_dataloader)
    res = {'frames': 0, 'feat_equal': 0, 'pos_equal': 0}
    with torch.no_grad():
        for data in dl:
            if res['frames'] >= n:
                break
            model.test_step(data)
            fe, pe = head.debug_gather_equal
            res['frames'] += len(data['data_samples'])
            res['feat_equal'] += int(fe) * len(data['data_samples'])
            res['pos_equal'] += int(pe) * len(data['data_samples'])
    head.debug_check_gather = False
    res['ok'] = res['frames'] > 0 and \
        res['feat_equal'] == res['pos_equal'] == res['frames']
    return res


def main():
    from mmengine.config import DictAction
    p = argparse.ArgumentParser(description='exec doc #3 evaluation')
    p.add_argument('config')
    p.add_argument('checkpoint')
    p.add_argument('--tag', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--ref-json', required=True,
                   help='exec doc #2 E0 oracle json')
    p.add_argument('--tol', type=float, default=None)
    p.add_argument('--debug-frames', type=int, default=0)
    p.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = p.parse_args()
    from mmengine.runner import Runner
    dc = qc.diag_common()
    t0 = time.time()
    cfg = qc.load_cfg(args.config, args.cfg_options)
    run_dir = osp.join(args.out_dir, 'oracle_' + args.tag)
    os.makedirs(run_dir, exist_ok=True)
    cfg.work_dir = run_dir
    cfg.launcher = 'none'
    cfg.load_from = None
    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(args.checkpoint, map_location='cpu')
    dbg = debug_frames(runner, cfg, args.debug_frames) \
        if args.debug_frames > 0 else None
    metrics = runner.test()
    metrics = {k: (float(v) if isinstance(v, (int, float)) else v)
               for k, v in metrics.items()}
    head = qc.unwrap(runner.model).bbox_head
    out = {'args': {'tag': args.tag, 'config': args.config,
                    'checkpoint': args.checkpoint, 'oracle': 'none',
                    'range': [0, 1000], 'score': 'orig',
                    'query_init': getattr(head, 'query_init', 'cell'),
                    'feat_sampling': getattr(head, 'feat_sampling', None),
                    'cfg_options': args.cfg_options},
           'stats': {'injected_cells': 0, 'debug': dbg},
           'metrics': metrics,
           'runtime': dc.runtime_info(cfg, args.config, args.checkpoint, t0,
                                      args.cfg_options,
                                      extra={'queryloc_md5': qc.ql_md5()})}
    path = osp.join(run_dir, 'oracle_%s.json' % args.tag)
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    with open(args.ref_json) as f:
        ref = json.load(f)['metrics']
    dm = metrics[qc.PRE + 'mAP'] - float(ref[qc.PRE + 'mAP'])
    dn = metrics[qc.PRE + 'NDS'] - float(ref[qc.PRE + 'NDS'])
    L = ['Exec doc #3 evaluation %s (%s, query_init=%s, feat_sampling=%s)'
         % (args.tag, osp.basename(args.config), out['args']['query_init'],
            out['args']['feat_sampling']),
         '  mAP %.6f  NDS %.6f  | vs E0: dmAP %+.6f (%+.3f pp)  dNDS %+.6f '
         '(%+.3f pp)' % (metrics[qc.PRE + 'mAP'], metrics[qc.PRE + 'NDS'],
                         dm, 100 * dm, dn, 100 * dn)]
    checks = []
    if dbg is not None:
        checks.append(('S-M0 bitwise (%d frames)' % dbg['frames'], dbg,
                       dbg['ok']))
    if args.tol is not None:
        checks.append(('S-M0 |dmAP|, |dNDS| <= %g vs E0' % args.tol,
                       {'dmAP': dm, 'dNDS': dn},
                       abs(dm) <= args.tol + 1e-12 and
                       abs(dn) <= args.tol + 1e-12))
    for name, val, ok in checks:
        L.append('  %-36s %s  %s' % (name, 'PASS' if ok else 'FAIL',
                                     json.dumps(val)))
    L.append('')
    try:
        L += qc.breakdown(args.ref_json, path)
    except (SystemExit, Exception) as e:  # noqa: B902
        L.append('breakdown unavailable: %r' % (e, ))
    ok = all(c[2] for c in checks)
    res = {'tag': args.tag, 'json': path, 'dmAP': dm, 'dNDS': dn,
           'checks': [{'name': n, 'value': v, 'result':
                       'PASS' if o else 'FAIL'} for n, v, o in checks],
           'result': 'PASS' if ok else 'FAIL',
           'elapsed_s': round(time.time() - t0, 1)}
    qc.write_report(args.out_dir, 'eval_' + args.tag, L, res)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
