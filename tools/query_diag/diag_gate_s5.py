#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S5 gate and S5 adjudication for the oracle runs (exec doc v1.1, sec. 8).

gate        Compare the hooked 'base' run (oracle_base.json) with the known
            BASE_MAP / BASE_NDS. Tolerance 0.001 on each.
            Exit 0 = S5 PASS, 1 = S5 FAIL.
adjudicate  Compare the hooked 'base' run with an un-hooked tools/test.py run
            of the same CONFIG / CKPT. Tolerance 0.0005 on each.
            Exit 0 = verdict SOURCE: the hook does not change inference, the
                     known values are the problem. The un-hooked numbers are
                     only a CANDIDATE reference; they must be confirmed by
                     the reviewer before anything else runs.
            Exit 1 = verdict HOOK: the hook changes inference; stop.

Pure json / text parsing: no GPU, no torch, no mmengine.

Usage:
  python diag_gate_s5.py gate --base-json B.json --map 0.6651 --nds 0.6962
  python diag_gate_s5.py adjudicate --base-json B.json --nohook-dir DIR
"""
import argparse
import glob
import hashlib
import json
import os
import os.path as osp
import re
import sys
import time

# Float slack so that e.g. |0.6650 - 0.6660| (= 0.0010000000000000009 in
# binary floating point) still counts as "<= 0.001".
EPS = 1e-9


def pick(metrics, suffix):
    for k, v in metrics.items():
        if k.endswith(suffix):
            try:
                return float(v)
            except (TypeError, ValueError):
                return float('nan')
    return float('nan')


def within(a, b, tol):
    return abs(a - b) <= tol + EPS


def file_md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_version_line(path='VERSION'):
    try:
        with open(path) as f:
            return f.readline().strip() or 'unknown'
    except (OSError, UnicodeDecodeError):
        return 'unknown'


def load_base(path):
    with open(path) as f:
        r = json.load(f)
    if r.get('args', {}).get('oracle') != 'none':
        sys.exit('[S5] %s is not a base (--oracle none) run' % path)
    m, n = pick(r['metrics'], 'mAP'), pick(r['metrics'], 'NDS')
    if m != m or n != n:
        sys.exit('[S5] no mAP / NDS in %s' % path)
    return r, m, n


def find_nohook_metrics(work_dir):
    """Metrics of the newest tools/test.py run under work_dir.

    mmengine's LoggerHook.after_test_epoch dumps one json object with the
    full-precision metrics to <work_dir>/<timestamp>/<timestamp>.json.
    Falls back to the text log (4 decimals) if no such json is found.
    """
    cands = sorted(glob.glob(osp.join(work_dir, '*', '*.json')),
                   key=osp.getmtime)
    for p in reversed(cands):
        try:
            with open(p) as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        if isinstance(d, dict):
            m, n = pick(d, 'mAP'), pick(d, 'NDS')
            if m == m and n == n:
                return p, m, n
    logs = sorted(glob.glob(osp.join(work_dir, '*', '*.log')),
                  key=osp.getmtime)
    for p in reversed(logs):
        with open(p, errors='replace') as f:
            txt = f.read()
        mm = re.findall(r'/mAP: ([0-9.]+)', txt)
        nn = re.findall(r'/NDS: ([0-9.]+)', txt)
        if mm and nn:
            return p + ' (text log, 4 decimals)', float(mm[-1]), float(nn[-1])
    sys.exit('[S5] no test metrics found under %s' % work_dir)


def write_json(path, res):
    if path:
        d = osp.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(res, f, indent=2)
        print('[S5] saved', path)


def runtime(args):
    rt = {'time': time.strftime('%Y-%m-%d %H:%M:%S'),
          'repo_version': read_version_line()}
    for key in ('config', 'ckpt'):
        p = getattr(args, key, None)
        if p:
            rt[key] = p
            rt[key + '_md5'] = file_md5(p) if osp.isfile(p) else 'missing'
    return rt


def cmd_gate(args):
    _, m, n = load_base(args.base_json)
    ok_m, ok_n = within(m, args.map, args.tol), within(n, args.nds, args.tol)
    ok = ok_m and ok_n
    print('[S5] base(hooked) mAP=%.4f NDS=%.4f | known mAP=%.4f NDS=%.4f | '
          'diff mAP=%+.4f NDS=%+.4f | tol %.4f'
          % (m, n, args.map, args.nds, m - args.map, n - args.nds, args.tol))
    print('[S5] %s' % ('PASS' if ok else 'FAIL'))
    if not ok:
        print('[S5] Pre-registered procedure (exec doc v1.1 sec. 8): do NOT '
              'run the other four groups; run\n'
              '       bash tools/query_diag/run_diag.sh s5-adjudicate\n'
              '     (un-hooked tools/test.py, 1 GPU) and report both results. '
              'The known values must not be replaced without confirmation.')
    write_json(args.out, {
        'check': 'S5', 'result': 'PASS' if ok else 'FAIL', 'tol': args.tol,
        'base_json': args.base_json,
        'base': {'mAP': m, 'NDS': n},
        'known': {'mAP': args.map, 'NDS': args.nds},
        'diff': {'mAP': m - args.map, 'NDS': n - args.nds},
        'runtime': runtime(args)})
    return 0 if ok else 1


def cmd_adjudicate(args):
    _, bm, bn = load_base(args.base_json)
    src, um, un = find_nohook_metrics(args.nohook_dir)
    same = within(bm, um, args.tol) and within(bn, un, args.tol)
    verdict = 'SOURCE' if same else 'HOOK'
    print('[S5-adj] hooked base : mAP=%.4f NDS=%.4f (%s)'
          % (bm, bn, args.base_json))
    print('[S5-adj] un-hooked   : mAP=%.4f NDS=%.4f (%s)' % (um, un, src))
    print('[S5-adj] diff        : mAP=%+.5f NDS=%+.5f | tol %.4f'
          % (bm - um, bn - un, args.tol))
    res = {'check': 'S5-adjudication', 'verdict': verdict, 'tol': args.tol,
           'hooked_base': {'mAP': bm, 'NDS': bn, 'source': args.base_json},
           'unhooked': {'mAP': um, 'NDS': un, 'source': src},
           'diff': {'mAP': bm - um, 'NDS': bn - un}}
    if args.map is not None and args.nds is not None:
        res['known'] = {'mAP': args.map, 'NDS': args.nds}
        print('[S5-adj] known       : mAP=%.4f NDS=%.4f | un-hooked - known: '
              'mAP=%+.4f NDS=%+.4f'
              % (args.map, args.nds, um - args.map, un - args.nds))
    if same:
        print('[S5-adj] verdict SOURCE: the hook does not change inference; '
              'the known values do not match this evaluation setup.\n'
              '         The un-hooked numbers are a CANDIDATE reference only. '
              'STOP and report both results; continue only after the '
              'reviewer confirms the new reference.')
    else:
        print('[S5-adj] verdict HOOK: the hook changes inference results. '
              'STOP and report.')
    res['runtime'] = runtime(args)
    write_json(args.out, res)
    return 0 if same else 1


def main():
    p = argparse.ArgumentParser(description='S5 gate / adjudication')
    sub = p.add_subparsers(dest='cmd')
    g = sub.add_parser('gate')
    g.add_argument('--base-json', required=True)
    g.add_argument('--map', type=float, required=True)
    g.add_argument('--nds', type=float, required=True)
    g.add_argument('--tol', type=float, default=0.001,
                   help='pre-registered: 0.001 (do not change)')
    g.add_argument('--config')
    g.add_argument('--ckpt')
    g.add_argument('--out')
    a = sub.add_parser('adjudicate')
    a.add_argument('--base-json', required=True)
    a.add_argument('--nohook-dir', required=True)
    a.add_argument('--map', type=float)
    a.add_argument('--nds', type=float)
    a.add_argument('--tol', type=float, default=0.0005,
                   help='pre-registered: 0.0005 (do not change)')
    a.add_argument('--config')
    a.add_argument('--ckpt')
    a.add_argument('--out')
    args = p.parse_args()
    if args.cmd == 'gate':
        return cmd_gate(args)
    if args.cmd == 'adjudicate':
        return cmd_adjudicate(args)
    p.print_help()
    return 2


if __name__ == '__main__':
    sys.exit(main())
