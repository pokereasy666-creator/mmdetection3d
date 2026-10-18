#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Where does an oracle mAP gain come from?

Reads oracle_*.json files (recursively under a directory) and prints,
relative to the base run (tag given by --base, default 'base'):
  - class-mean AP at each nuScenes match threshold (0.5 / 1 / 2 / 4 m):
    gains at the tight 0.5 m threshold mean better localization of objects
    that are found anyway; gains at 2 / 4 m mean more objects found
  - per-class AP (mean over the 4 thresholds)
  - TP error metrics (mATE ... mAAE) and NDS
  - the near/far interaction: dmAP(all) - dmAP(near) - dmAP(far)
Pure json, no torch. Usage: python oracle_breakdown.py <dir> [--base TAG]
"""
import argparse
import glob
import json
import os.path as osp
import sys

PRE = 'NuScenes metric/pred_instances_3d_NuScenes/'
CLASSES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
           'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
THRS = ['0.5', '1.0', '2.0', '4.0']
ORDER = ['base', 'all', 'near', 'far', 'all_cls']


def breakdown_lines(root, base='base'):
    """Return the report lines (also used by diag_exec2_check.py)."""
    L = []

    def print(s=''):  # noqa: A001
        L.append(s)

    runs = {}
    for f in glob.glob(osp.join(root, '**', 'oracle_*.json'), recursive=True):
        with open(f) as fh:
            r = json.load(fh)
        runs[r['args']['tag']] = r['metrics']
    if base not in runs:
        raise SystemExit('no run tagged %r under %s' % (base, root))
    tags = [base] + [t for t in ORDER if t in runs and t != base] + \
        sorted(t for t in runs if t not in ORDER and t != base)
    w = max([8] + [len(t) for t in tags])  # 8 for the exec doc #1 tags

    def g(t, k):
        return float(runs[t][PRE + k])

    def ap(t, cls, th):
        return g(t, '%s_AP_dist_%s' % (cls, th))

    print('Class-mean AP per match threshold, pp (delta vs base)')
    print('%-*s' % (w, 'run') + ''.join('%16s' % ('AP@' + th + 'm')
                                        for th in THRS) + '%9s' % 'mAP')
    for t in tags:
        cells = []
        for th in THRS:
            v = 100 * sum(ap(t, c, th) for c in CLASSES) / len(CLASSES)
            b = 100 * sum(ap(base, c, th) for c in CLASSES) / len(CLASSES)
            cells.append('%7.2f (%+5.2f)' % (v, v - b))
        print('%-*s' % (w, t) + ''.join('%16s' % c for c in cells) +
              '%9.3f' % (100 * g(t, 'mAP')))
    print('\nShare of dmAP by threshold (dmAP = mean over thresholds), pp')
    for t in tags:
        if t == base:
            continue
        parts = [100 * sum(ap(t, c, th) - ap(base, c, th)
                           for c in CLASSES) / len(CLASSES) / len(THRS)
                 for th in THRS]
        print('%-*s ' % (w, t) + '  '.join('@%sm %+6.3f' % (th, p)
                                           for th, p in zip(THRS, parts)) +
              '  | dmAP %+6.3f' % (100 * (g(t, 'mAP') - g(base, 'mAP'))))
    print('\nPer-class AP (mean over thresholds), pp: base | delta per run')
    print('%-22s %7s | ' % ('class', 'base') +
          ' '.join('%*s' % (w, t) for t in tags if t != base))
    for c in CLASSES:
        def m(t):
            return 100 * sum(ap(t, c, th) for th in THRS) / len(THRS)
        print('%-22s %7.2f | ' % (c, m(base)) +
              ' '.join('%+*.2f' % (w, m(t) - m(base))
                       for t in tags if t != base))
    print('\nTP errors and NDS')
    print('%-6s' % '' + ''.join('%*s' % (w + 1, t) for t in tags))
    for k in ('mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE', 'NDS'):
        print('%-6s' % k + ''.join('%*.4f' % (w + 1, g(t, k))
                                   for t in tags))
    if all(t in runs for t in ('all', 'near', 'far')):
        d = {t: 100 * (g(t, 'mAP') - g(base, 'mAP'))
             for t in ('all', 'near', 'far')}
        print('\nInteraction dmAP(all) - dmAP(near) - dmAP(far) = %+.3f pp'
              % (d['all'] - d['near'] - d['far']))
    return L


def main():
    p = argparse.ArgumentParser(description='oracle gain breakdown')
    p.add_argument('root')
    p.add_argument('--base', default='base', help='tag of the base run')
    args = p.parse_args()
    for line in breakdown_lines(args.root, args.base):
        sys.stdout.write(line + '\n')


if __name__ == '__main__':
    main()
