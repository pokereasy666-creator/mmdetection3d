#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collect oracle_*.json files and print mAP / NDS deltas vs. the baseline.

Usage: python diag_summarize.py work_dirs/diag [--base base]
"""
import argparse
import glob
import json
import os.path as osp


def pick(metrics, suffix):
    for k, v in metrics.items():
        if k.endswith(suffix):
            try:
                return float(v)
            except ValueError:
                return float('nan')
    return float('nan')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root')
    p.add_argument('--base', default='base')
    args = p.parse_args()

    runs = {}
    for f in sorted(glob.glob(osp.join(args.root, '**', 'oracle_*.json'),
                              recursive=True)):
        with open(f) as fh:
            r = json.load(fh)
        runs[r['args']['tag']] = r
    if not runs:
        print('no oracle_*.json found under', args.root)
        return
    base = runs.get(args.base)
    bm = pick(base['metrics'], 'mAP') if base else float('nan')
    bn = pick(base['metrics'], 'NDS') if base else float('nan')
    print('%-10s %-8s %-12s %-6s %8s %8s %8s %8s %10s'
          % ('tag', 'oracle', 'range', 'score', 'mAP', 'dmAP', 'NDS', 'dNDS',
             'injected'))
    for tag, r in runs.items():
        a = r['args']
        m, n = pick(r['metrics'], 'mAP'), pick(r['metrics'], 'NDS')
        print('%-10s %-8s %-12s %-6s %8.4f %+8.4f %8.4f %+8.4f %10d'
              % (tag, a['oracle'], '%g-%g' % tuple(a['range']), a['score'],
                 m, m - bm, n, n - bn, int(r['stats']['injected_cells'])))

    rec = glob.glob(osp.join(args.root, '**', 'recall_summary.json'),
                    recursive=True)
    for f in rec:
        with open(f) as fh:
            s = json.load(fh)
        print('\n[recall] %s  frames=%s  init-label match=%.5f'
              % (f, s.get('frames'), s.get('init_label_match_rate', -1)))
        t1 = s.get('T1_class-aware', {})
        thr = s.get('thr', [])
        for b, v in t1.items():
            print('  %-8s N=%6d  ' % (b, v['n']) +
                  '  '.join('@%gm %.3f' % (t, x)
                            for t, x in zip(thr, v['recall'])))


if __name__ == '__main__':
    main()
