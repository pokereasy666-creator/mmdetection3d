#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B0 (exec doc #2): how much of the initial-position error does the decoder
correct? Offline, from the D1 per-GT arrays of exec doc #1 step 2.

Reads   work_dirs/diag/recall/recall_per_gt.npz (diag_query_recall.py)
Writes  <out-dir>/b0_decoder.txt and <out-dir>/b0_decoder.json

All quantities: K = num_proposals (200), class-aware.
  B0-1  init vs final recall per bin and threshold (0.5 / 1 / 2 / 4 m)
  B0-2  GT grouped by the distance of the nearest same-class initial query
        (G0 <= 0.5 m, G1 (0.5, 1] m, G2 (1, 2] m) and the share whose final
        (decoded) center falls within a tighter threshold. Approximate: the
        nearest initial query need not be the one that produced the final
        center.
  N_missed_D1  GT with rank_ca(2 m) >= K, the reference for S8
  S9    R_near(K), R_far(K) at 2 m recomputed from the npz must equal the
        exec doc #1 step 2 values to 5 decimals (exit 1 otherwise)
No GPU, no torch. Usage:
  python diag_b0_decoder.py --npz work_dirs/diag/recall/recall_per_gt.npz \
      --out-dir work_dirs/diag/exec2
"""
import argparse
import json
import os
import os.path as osp
import sys
import time

import numpy as np

BINS = (('0-20', 0.0, 20.0), ('20-30', 20.0, 30.0), ('30-40', 30.0, 40.0),
        ('40-50', 40.0, 50.0), ('<30', 0.0, 30.0), ('>=30', 30.0, 1e9),
        ('all', 0.0, 1e9))
GROUP_BINS = ('<30', '>=30', 'all')


def rate(mask):
    return float(np.mean(mask)) if mask.size else float('nan')


def read_version_line(path='VERSION'):
    try:
        with open(path) as f:
            return f.readline().strip() or 'unknown'
    except (OSError, UnicodeDecodeError):
        return 'unknown'


def main():
    p = argparse.ArgumentParser(description='B0 decoder correction (exec #2)')
    p.add_argument('--npz', default='work_dirs/diag/recall/recall_per_gt.npz')
    p.add_argument('--out-dir', default='work_dirs/diag/exec2')
    p.add_argument('--s9-near', default='0.97319',
                   help='pre-registered R_near(200) @2 m (exec doc #1 step 2)')
    p.add_argument('--s9-far', default='0.94944',
                   help='pre-registered R_far(200) @2 m (exec doc #1 step 2)')
    args = p.parse_args()

    Z = np.load(args.npz, allow_pickle=False)
    thr = [float(x) for x in Z['thr']]
    P = int(Z['P'])
    dist = Z['dist']
    rc = Z['rank_ca']
    fc = Z['fin_ca']
    j = {t: thr.index(t) for t in (0.5, 1.0, 2.0, 4.0)}
    init = {t: rc[:, j[t]] < P for t in j}          # class-aware, K = P
    fin = {t: fc[:, j[t]] for t in j}
    masks = {name: (dist >= lo) & (dist < hi) for name, lo, hi in BINS}

    L = []
    res = {'npz': args.npz, 'P': P, 'num_gt': int(dist.size),
           'class_range': bool(Z['class_range'])
           if 'class_range' in Z.files else 'unknown'}

    # B0-1
    L.append('B0-1 init vs final recall (class-aware, K=%d); diff = final '
             '- init' % P)
    L.append('%-6s %7s ' % ('bin', 'N_gt') + ' '.join(
        '%23s' % ('@%gm init/final/diff' % t) for t in (0.5, 1, 2, 4)))
    b01 = {}
    for name, _, _ in BINS:
        m = masks[name]
        row = {}
        cells = []
        for t in (0.5, 1.0, 2.0, 4.0):
            a, b = rate(init[t][m]), rate(fin[t][m])
            row['%g' % t] = {'init': a, 'final': b, 'diff': b - a}
            cells.append('%7.4f/%7.4f/%+7.4f' % (a, b, b - a))
        b01[name] = {'n': int(m.sum()), 'thr': row}
        L.append('%-6s %7d ' % (name, m.sum()) +
                 ' '.join('%23s' % c for c in cells))
    res['B0_1'] = b01

    # B0-2
    g0 = init[0.5]
    g1 = ~init[0.5] & init[1.0]
    g2 = ~init[1.0] & init[2.0]
    L.append('')
    L.append('B0-2 decoder correction by nearest same-class initial query '
             '(approximate)')
    L.append('%-6s %8s %10s | %8s %10s | %8s %10s %10s'
             % ('bin', 'N_G0', 'G0 keep@.5', 'N_G1', 'G1 fix@.5', 'N_G2',
                'G2 to@.5', 'G2 to@1'))
    b02 = {}
    for name in GROUP_BINS:
        m = masks[name]
        r = {'G0': {'n': int((g0 & m).sum()),
                    'keep_0.5': rate(fin[0.5][g0 & m])},
             'G1': {'n': int((g1 & m).sum()),
                    'fix_0.5': rate(fin[0.5][g1 & m])},
             'G2': {'n': int((g2 & m).sum()),
                    'to_0.5': rate(fin[0.5][g2 & m]),
                    'to_1.0': rate(fin[1.0][g2 & m])}}
        b02[name] = r
        L.append('%-6s %8d %10.4f | %8d %10.4f | %8d %10.4f %10.4f'
                 % (name, r['G0']['n'], r['G0']['keep_0.5'], r['G1']['n'],
                    r['G1']['fix_0.5'], r['G2']['n'], r['G2']['to_0.5'],
                    r['G2']['to_1.0']))
    res['B0_2'] = b02

    # N_missed_D1 and S9
    missed = ~init[2.0]
    nm = {name: int((missed & masks[name]).sum()) for name in GROUP_BINS}
    res['N_missed_D1'] = nm
    near = '%.5f' % rate(init[2.0][masks['<30']])
    far = '%.5f' % rate(init[2.0][masks['>=30']])
    s9 = near == args.s9_near and far == args.s9_far
    res['S9'] = {'R_near': near, 'R_far': far,
                 'expected': {'R_near': args.s9_near, 'R_far': args.s9_far},
                 'result': 'PASS' if s9 else 'FAIL'}
    L.append('')
    L.append('N_missed_D1 (rank_ca @2m >= %d): <30 %d | >=30 %d | all %d'
             % (P, nm['<30'], nm['>=30'], nm['all']))
    L.append('S9  R_near(%d)=%s (expected %s)  R_far(%d)=%s (expected %s)  %s'
             % (P, near, args.s9_near, P, far, args.s9_far,
                res['S9']['result']))
    res['runtime'] = {'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                      'repo_version': read_version_line()}

    text = '\n'.join(L)
    print(text)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(osp.join(args.out_dir, 'b0_decoder.txt'), 'w') as f:
        f.write(text + '\n')
    with open(osp.join(args.out_dir, 'b0_decoder.json'), 'w') as f:
        json.dump(res, f, indent=2)
    print('[b0] saved %s/b0_decoder.{txt,json}' % args.out_dir)
    return 0 if s9 else 1


if __name__ == '__main__':
    sys.exit(main())
