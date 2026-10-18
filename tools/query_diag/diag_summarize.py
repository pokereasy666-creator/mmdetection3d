#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collect oracle_*.json files and print mAP / NDS deltas vs. the baseline.

Usage: python diag_summarize.py work_dirs/diag [--base base]
Also writes <root>/summary.txt with the same text.
"""
import argparse
import builtins
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
    lines = []

    def print(*a):  # noqa: A001  echo to the terminal and keep for the file
        s = ' '.join(str(x) for x in a)
        lines.append(s)
        builtins.print(s)

    try:
        _main(args, print)
    finally:
        path = osp.join(args.root, 'summary.txt')
        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        builtins.print('[summary] saved', path)


def _main(args, print):
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

    # Self-checks S5-S7 (exec doc v1.1 sec. 8). R-rule verdicts are left
    # to the review on purpose.
    print('\nSelf-checks')
    gates = glob.glob(osp.join(args.root, '**', 's5_gate.json'),
                      recursive=True)
    if gates:
        with open(max(gates, key=osp.getmtime)) as fh:
            g = json.load(fh)
        print('  S5  base vs known: dmAP=%+.4f dNDS=%+.4f (tol %.4f)  %s'
              % (g['diff']['mAP'], g['diff']['NDS'], g['tol'], g['result']))
    else:
        print('  S5  no s5_gate.json found  MISSING')
    for tag, r in runs.items():
        if r['args']['oracle'] == 'none':
            continue
        st = r['stats']
        s6 = st['init_mismatch_rate'] <= 0.01 and st['capped_cells'] == 0
        s7 = st['restored_queries'] >= 0.99 * st['injected_cells']
        print('  S6  %-8s init_mismatch_rate=%.5f (<= 0.01) capped_cells=%d '
              '(== 0)  %s' % (tag, st['init_mismatch_rate'],
                              int(st['capped_cells']),
                              'PASS' if s6 else 'FAIL'))
        print('  S7  %-8s restored_queries=%d >= 0.99 x injected_cells=%d  '
              '%s' % (tag, int(st['restored_queries']),
                      int(st['injected_cells']), 'PASS' if s7 else 'FAIL'))
    for tag, r in runs.items():
        opts = r.get('runtime', {}).get('cfg_options') or {}
        if opts:
            print('  NOTE %-8s ran with cfg_options=%s' % (tag, opts))

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
