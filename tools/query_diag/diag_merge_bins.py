#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Merged "<30 m" / ">=30 m" bins and all rule inputs from a recall run.

Reads   <dir>/recall_per_gt.npz, recall_summary.json, recall_report.txt
        (written by diag_query_recall.py) and does NOT change any of the
        statistics computed there; it only re-aggregates the per-GT arrays.
Writes  <dir>/recall_merged.json and <dir>/recall_merged.txt

Self-checks S1, S2, S3, S4a, S4 get PASS / FAIL. The R-rules of exec doc
v1.1 sec. 10 are intentionally NOT evaluated here: only their input
quantities are printed (sec. 7 list); the verdicts belong to the review.

Usage: python diag_merge_bins.py work_dirs/diag/recall [more dirs ...]
"""
import argparse
import json
import os.path as osp
import sys
import time

import numpy as np

FRAC_OVER = 200          # sec. 7: frames with more than 200 valid GT


def rate(mask):
    return float(np.mean(mask)) if mask.size else float('nan')


def read_version_line(path='VERSION'):
    try:
        with open(path) as f:
            return f.readline().strip() or 'unknown'
    except (OSError, UnicodeDecodeError):
        return 'unknown'


def bin_stats(D, m, ks, ti, P):
    """class-aware / class-agnostic init recall for every K, final recall."""
    n = int(m.sum())
    rc, ra = D['rank_ca'], D['rank_agn']
    out = {'n_gt': n,
           'ca': {str(k): rate(rc[m, ti] < k) for k in ks},
           'agn': {str(k): rate(ra[m, ti] < k) for k in ks},
           'final_ca': rate(D['fin_ca'][m, ti]),
           'final_agn': rate(D['fin_agn'][m, ti])}
    out['init_ca_P'] = out['ca'][str(P)]
    out['init_agn_P'] = out['agn'][str(P)]
    return out


def check(name, desc, value, ok, crit):
    if value is None or (isinstance(value, float) and value != value):
        res = 'MISSING'
    else:
        res = 'PASS' if ok else 'FAIL'
    return {'id': name, 'desc': desc, 'value': value, 'criterion': crit,
            'result': res}


def ratio(a, b):
    if b == 0:
        return float('inf') if a > 0 else float('nan')
    return a / b


def merge_one(d, split):
    npz = osp.join(d, 'recall_per_gt.npz')
    sj = osp.join(d, 'recall_summary.json')
    rp = osp.join(d, 'recall_report.txt')
    Z = np.load(npz, allow_pickle=False)
    with open(sj) as f:
        S = json.load(f)
    with open(rp) as f:
        report = f.read()

    thr = [float(x) for x in Z['thr']]
    ks = [int(x) for x in Z['ks']]
    P = int(Z['P'])
    main_thr = float(S.get('main_thr', 2.0))
    ti = thr.index(main_thr)
    k_hi = max(ks)
    D = {k: Z[k] for k in ('dist', 'rank_ca', 'rank_agn', 'fin_ca',
                           'fin_agn')}
    n = int(D['dist'].shape[0])
    if n == 0:
        sys.exit('[merge] %s has no GT' % npz)
    dist = D['dist']
    fc = Z['frame_counts']
    class_range = bool(Z['class_range']) if 'class_range' in Z.files \
        else S.get('class_range', 'unknown')

    bins = {
        '<%g m' % split: dist < split,
        '>=%g m' % split: dist >= split,
        '0-20 m': (dist >= 0) & (dist < 20),
        'all': np.ones(n, dtype=bool),
    }
    B = {name: bin_stats(D, m, ks, ti, P) for name, m in bins.items()}
    near, far = B['<%g m' % split], B['>=%g m' % split]

    r_near_p, r_far_p = near['ca'][str(P)], far['ca'][str(P)]
    r_near_hi, r_far_hi = near['ca'][str(k_hi)], far['ca'][str(k_hi)]
    miss_near, miss_far = 1.0 - r_near_p, 1.0 - r_far_p
    rule_inputs = {
        'R_near(%d)' % P: r_near_p,
        'R_near(%d)' % k_hi: r_near_hi,
        'R_far(%d)' % P: r_far_p,
        'R_far(%d)' % k_hi: r_far_hi,
        'miss_near': miss_near,
        'miss_far': miss_far,
        'miss_far/miss_near': ratio(miss_far, miss_near),
        'dR_far = R_far(%d)-R_far(%d)' % (k_hi, P): r_far_hi - r_far_p,
        'agn_gap_far = agnR_far(%d)-R_far(%d)' % (P, P):
            far['agn'][str(P)] - r_far_p,
        'final_gap_far = final_ca_far-R_far(%d)' % P:
            far['final_ca'] - r_far_p,
        'frac_over_%d' % FRAC_OVER: float(np.mean(fc > FRAC_OVER))
        if fc.size else float('nan'),
        'n_gt_near': near['n_gt'],
        'n_gt_far': far['n_gt'],
        'frames': int(fc.size),
    }

    near20 = B['0-20 m']
    n_warn = report.count('[WARNING]')
    match = S.get('init_label_match_rate')
    transposed = S.get('bev_pos_transposed', None)
    cg = S.get('center_grid_in_range_frac')
    s4_delta = near20['final_agn'] - near20['init_agn_P']
    s3_val = near20['init_ca_P']
    checks = [
        check('S1', 'init vs head.query_labels match rate', match,
              match is not None and match >= 0.99, '>= 0.99'),
        check('S2', 'bev_pos transposed', transposed,
              transposed is False, '== False'),
        check('S3a', 'number of [WARNING] lines in recall_report.txt',
              n_warn, n_warn == 0, '== 0'),
        check('S3b', 'class-aware init recall @%gm K=%d, 0-20 m'
              % (main_thr, P), s3_val, s3_val >= 0.5, '>= 0.5'),
        check('S4a', 'center_grid values within [-2, H+2] (grid units)',
              cg, cg is not None and cg >= 0.99, '>= 0.99'),
        check('S4', '0-20 m final_agn - init_agn @%gm' % main_thr,
              s4_delta, s4_delta >= -0.05, '>= -0.05'),
    ]
    s3 = 'PASS' if all(c['result'] == 'PASS' for c in checks
                       if c['id'] in ('S3a', 'S3b')) else 'FAIL'
    all_pass = all(c['result'] == 'PASS' for c in checks)

    res = {'source_dir': d, 'frames': S.get('frames'), 'num_gt': n,
           'class_range': class_range, 'main_thr': main_thr, 'P': P,
           'ks': ks, 'split_m': split,
           'self_checks': checks, 'S3': s3,
           'self_checks_all_pass': all_pass,
           'rule_inputs': rule_inputs,
           'bins': B,
           'note': 'R-rule verdicts are intentionally not computed; the '
                   'reviewer applies exec doc v1.1 sec. 10.',
           'runtime': {'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                       'repo_version': read_version_line(),
                       'recall_runtime': S.get('runtime', {})}}

    L = []
    L.append('[merge] source=%s frames=%s num_gt=%d class_range=%s '
             'main_thr=%gm P=%d' % (d, S.get('frames'), n, class_range,
                                    main_thr, P))
    if P != FRAC_OVER:
        L.append('[merge] NOTE: num_proposals=%d differs from %d used for '
                 'frac_over' % (P, FRAC_OVER))
    L.append('')
    L.append('Self-checks')
    for c in checks:
        v = c['value']
        vs = ('%.5f' % v) if isinstance(v, float) else str(v)
        L.append('  %-4s %-52s %10s  %-9s %s'
                 % (c['id'], c['desc'], vs, c['criterion'], c['result']))
    L.append('  S3 overall: %s | all self-checks PASS: %s' % (s3, all_pass))
    L.append('')
    L.append('Bins @%gm (class-aware "ca", class-agnostic "agn"; initial '
             'recall per K; final = after decoder)' % main_thr)
    hdr = '  %-8s %7s ' % ('bin', 'N_gt') + \
        ' '.join('%8s' % ('ca@%d' % k) for k in ks) + ' ' + \
        ' '.join('%8s' % ('agn@%d' % k) for k in ks) + \
        ' %9s %9s' % ('final_ca', 'final_agn')
    L.append(hdr)
    for name, b in B.items():
        L.append('  %-8s %7d ' % (name, b['n_gt']) +
                 ' '.join('%8.4f' % b['ca'][str(k)] for k in ks) + ' ' +
                 ' '.join('%8.4f' % b['agn'][str(k)] for k in ks) +
                 ' %9.4f %9.4f' % (b['final_ca'], b['final_agn']))
    L.append('')
    L.append('Rule inputs (exec doc v1.1 sec. 7; no R-rule verdicts here)')
    for k, v in rule_inputs.items():
        vs = ('%.5f' % v) if isinstance(v, float) else str(v)
        L.append('  %-44s %s' % (k, vs))
    text = '\n'.join(L)
    print(text)
    with open(osp.join(d, 'recall_merged.txt'), 'w') as f:
        f.write(text + '\n')
    with open(osp.join(d, 'recall_merged.json'), 'w') as f:
        json.dump(res, f, indent=2, default=str)
    print('[merge] saved %s/recall_merged.{txt,json}' % d)
    return all_pass


def main():
    p = argparse.ArgumentParser(description='merged bins + rule inputs')
    p.add_argument('dirs', nargs='+', help='recall output dir(s)')
    p.add_argument('--split', type=float, default=30.0,
                   help='near/far split in meters (fixed at 30 by the doc)')
    args = p.parse_args()
    ok = True
    for d in args.dirs:
        ok = merge_one(d, args.split) and ok
        print()
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
