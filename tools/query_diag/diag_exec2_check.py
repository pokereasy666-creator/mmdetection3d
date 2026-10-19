#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Checks and summary for exec doc #2 (coverage vs localization).

s0p       S0': CONFIG / CKPT md5 and CFG_OPTS equal to exec doc #1 (read
          from #1's oracle_base.json runtime), tools/query_diag/ changed only
          in files covered by M1-M5 (plus the logging fix required by
          section 2), sync gate (ASCII, ast for py3.8, bash -n, json).
step1-ok  exit 0 only if S0' and S9 (b0_decoder.json) both PASS; run_diag.sh
          uses it to refuse starting E0-E3 otherwise.
summary   M5: E1-E3 vs E0 (dmAP / dNDS in pp), E3 retention = dmAP(E3) /
          1.875, S0', S9, S5', S6, S7', S8 PASS/FAIL, and the per-threshold /
          per-class breakdown (oracle_breakdown.py). Q-rule verdicts are
          intentionally NOT computed.
Pure json / stdlib (+ the sibling oracle_breakdown.py); no torch.
"""
import argparse
import ast
import glob
import hashlib
import json
import os
import os.path as osp
import subprocess
import sys
import time

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
import oracle_breakdown  # noqa: E402

# Files allowed to differ from the versions that produced exec doc #1
# (S0' scope: M1-M5 plus the #1 logging persistence, decisions E2-1 /
# E2-2 in projects/DomeDETR/docs/EXEC02_DECISIONS.md).
CHANGE_CATEGORY = {
    'diag_oracle_query.py': 'M1/M2',
    'diag_b0_decoder.py': 'M3 (new)',
    'run_diag.sh': 'M4 (+ #1 logging persistence, decision E2-1)',
    'exec02_env.sh': 'M4 (new)',
    'diag_exec2_check.py': 'M5 (new)',
    'oracle_breakdown.py': 'M5 (moved from projects/DomeDETR/tools, E2-2)',
    'diag_summarize.py': 'LOG (#1 logging persistence, decision E2-1)',
}
TAGS = ('e0_base', 'e1_missed_orig', 'e2_missed_boost', 'e3_jitter')
ORIG_TAGS = ('e1_missed_orig', 'e3_jitter')
PRE = 'NuScenes metric/pred_instances_3d_NuScenes/'


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


def load(path):
    with open(path) as f:
        return json.load(f)


def write(out_dir, stem, lines, res):
    os.makedirs(out_dir, exist_ok=True)
    text = '\n'.join(lines)
    print(text)
    with open(osp.join(out_dir, stem + '.txt'), 'w') as f:
        f.write(text + '\n')
    with open(osp.join(out_dir, stem + '.json'), 'w') as f:
        json.dump(res, f, indent=2, default=str)
    print('[exec2] saved %s/%s.{txt,json}' % (out_dir, stem))


def gate_one(path):
    """Sync gate for one file; returns an error string or ''."""
    with open(path, 'rb') as f:
        raw = f.read()
    if any(b >= 128 for b in raw):
        return 'non-ASCII'
    if path.endswith('.py'):
        try:
            ast.parse(raw.decode(), feature_version=(3, 8))
        except SyntaxError as e:
            return 'ast: %s' % e
    elif path.endswith('.sh'):
        r = subprocess.run(['bash', '-n', path], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
        if r.returncode:
            return 'bash -n: %s' % r.stdout.decode(errors='replace')
    elif path.endswith('.json'):
        try:
            json.loads(raw.decode())
        except ValueError as e:
            return 'json: %s' % e
    return ''


def cmd_s0p(args):
    ref = load(args.ref_json).get('runtime', {})
    L, fail = ['S0\' consistency with exec doc #1 (%s)' % args.ref_json], []
    cm, km = file_md5(args.config), file_md5(args.ckpt)
    opts = os.environ.get('CFG_OPTS', '')
    checks = [('CONFIG md5', cm, ref.get('config_md5')),
              ('CKPT md5', km, ref.get('checkpoint_md5')),
              ('CFG_OPTS', opts, ref.get('cfg_opts_env'))]
    for name, now, want in checks:
        ok = now == want
        L.append('  %-10s %s  (#1: %s)  %s' % (name, now, want,
                                               'OK' if ok else 'DIFF'))
        if not ok:
            fail.append('%s differs from exec doc #1' % name)
    ref_md5 = ref.get('diag_files_md5', {})
    cur = {f: file_md5(osp.join(HERE, f)) for f in sorted(os.listdir(HERE))
           if f.endswith(('.py', '.sh', '.json'))}
    changes = []
    L.append('  tools/query_diag vs exec doc #1:')
    for f in sorted(set(ref_md5) | set(cur)):
        if ref_md5.get(f) == cur.get(f):
            continue
        st = 'NEW' if f not in ref_md5 else ('GONE' if f not in cur
                                             else 'CHANGED')
        cat = CHANGE_CATEGORY.get(f, 'OTHER')
        changes.append({'file': f, 'status': st, 'category': cat,
                        'md5_1': ref_md5.get(f), 'md5_now': cur.get(f)})
        L.append('    %-26s %-8s %-8s -> %-8s  %s'
                 % (f, st, (ref_md5.get(f) or '-')[:8],
                    (cur.get(f) or '-')[:8], cat))
        if cat == 'OTHER':
            fail.append('%s %s is not covered by M1-M5 / logging' % (f, st))
    gates = {}
    for f in sorted(cur):
        err = gate_one(osp.join(HERE, f))
        gates[f] = err or 'OK'
        if err:
            fail.append('gate %s: %s' % (f, err))
    L.append('  sync gate: %s' % ('all OK' if all(
        v == 'OK' for v in gates.values()) else gates))
    res = {'check': "S0'", 'result': 'FAIL' if fail else 'PASS',
           'config_md5': cm, 'ckpt_md5': km, 'cfg_opts_env': opts,
           'changes': changes, 'diag_files_md5_now': cur, 'gates': gates,
           'fail': fail, 'runtime': {
               'time': time.strftime('%Y-%m-%d %H:%M:%S'),
               'repo_version': read_version_line()}}
    for m in fail:
        L.append('  FAIL ' + m)
    L.append("S0'  %s" % res['result'])
    write(args.out_dir, 's0p_check', L, res)
    return 0 if not fail else 1


def cmd_step1_ok(args):
    try:
        s0p = load(osp.join(args.root, 's0p_check.json'))['result']
        s9 = load(osp.join(args.root, 'b0_decoder.json'))['S9']['result']
    except (OSError, KeyError, ValueError) as e:
        print('[exec2] step 1 incomplete: %r' % e)
        return 1
    print("[exec2] step 1: S0'=%s S9=%s" % (s0p, s9))
    return 0 if s0p == s9 == 'PASS' else 1


def pick(metrics, key):
    v = metrics.get(PRE + key)
    return float(v) if v is not None else float('nan')


def check(cid, desc, ok, value):
    res = 'MISSING' if ok is None else ('PASS' if ok else 'FAIL')
    return {'id': cid, 'desc': desc, 'value': value, 'result': res}


def cmd_summary(args):
    root = args.root
    runs = {}
    for f in glob.glob(osp.join(root, '**', 'oracle_*.json'), recursive=True):
        r = load(f)
        runs[r['args']['tag']] = r
    L, checks = [], []
    e0 = runs.get('e0_base')
    L.append('Exec doc #2 summary (dmAP / dNDS in pp vs E0; no Q verdicts)')
    L.append('%-16s %-7s %-7s %-6s %3s %8s %8s %8s %8s %9s %9s %9s %9s'
             % ('tag', 'oracle', 'subset', 'score', 'jit', 'mAP', 'dmAP',
                'NDS', 'dNDS', 'injected', 'selected', 'restored',
                'missed'))
    rows = {}
    for t in TAGS:
        r = runs.get(t)
        if r is None:
            L.append('%-16s MISSING' % t)
            continue
        a, st, m = r['args'], r['stats'], r['metrics']
        mp, nd = pick(m, 'mAP'), pick(m, 'NDS')
        dm = 100 * (mp - pick(e0['metrics'], 'mAP')) if e0 else float('nan')
        dn = 100 * (nd - pick(e0['metrics'], 'NDS')) if e0 else float('nan')
        rows[t] = {'mAP': mp, 'NDS': nd, 'dmAP_pp': dm, 'dNDS_pp': dn,
                   'stats': st, 'args': {k: a.get(k) for k in (
                       'oracle', 'subset', 'miss_thr', 'range', 'score',
                       'jitter_cells', 'seed')},
                   'runtime_elapsed_s': r.get('runtime', {}).get('elapsed_s')}
        L.append('%-16s %-7s %-7s %-6s %3s %8.4f %+8.3f %8.4f %+8.3f %9d %9d '
                 '%9d %9d' % (t, a['oracle'], a.get('subset', 'all'),
                              a['score'], a.get('jitter_cells', 0), mp, dm,
                              nd, dn, st.get('injected_cells', 0),
                              st.get('selected_injected', 0),
                              st.get('restored_queries', 0),
                              st.get('missed_gt', 0)))
    retention = None
    if 'e3_jitter' in rows:
        retention = rows['e3_jitter']['dmAP_pp'] / args.ref_dmap_all
    L.append('')
    L.append('coverage_orig  = dmAP(E1) = %s pp' % (
        '%+.3f' % rows['e1_missed_orig']['dmAP_pp']
        if 'e1_missed_orig' in rows else 'n/a'))
    L.append('coverage_boost = dmAP(E2) = %s pp' % (
        '%+.3f' % rows['e2_missed_boost']['dmAP_pp']
        if 'e2_missed_boost' in rows else 'n/a'))
    L.append('jitter retention = dmAP(E3) / %.3f = %s' % (
        args.ref_dmap_all, '%.3f' % retention if retention is not None
        else 'n/a'))

    def jget(path, *keys):
        try:
            v = load(path)
            for k in keys:
                v = v[k]
            return v
        except (OSError, KeyError, ValueError):
            return None

    s0p = jget(osp.join(root, 's0p_check.json'), 'result')
    s9 = jget(osp.join(root, 'b0_decoder.json'), 'S9', 'result')
    nmd1 = jget(osp.join(root, 'b0_decoder.json'), 'N_missed_D1', 'all')
    s5 = jget(osp.join(root, 'oracle_e0_base', 's5p_gate.json'))
    checks.append(check("S0'", 'consistency with exec doc #1',
                        None if s0p is None else s0p == 'PASS', s0p))
    checks.append(check('S9', 'R_near/R_far(200) @2m == #1 step 2',
                        None if s9 is None else s9 == 'PASS', s9))
    checks.append(check("S5'", 'E0 vs reference base (tol 0.001)',
                        None if s5 is None else s5['result'] == 'PASS',
                        None if s5 is None else s5['diff']))
    k6 = ('init_mismatch_rate', 'capped_cells')
    k7 = ('selected_injected', 'injected_cells', 'restored_queries')
    for t in TAGS[1:]:
        st = rows.get(t, {}).get('stats') or {}
        v6 = {k: st.get(k) for k in k6}
        v7 = {k: st.get(k) for k in k7}
        ok6 = None if None in v6.values() else (
            v6['init_mismatch_rate'] <= 0.01 and v6['capped_cells'] == 0)
        ok7 = None
        if None not in v7.values():
            ok7 = v7['selected_injected'] >= 0.99 * v7['injected_cells']
            if t in ORIG_TAGS:
                ok7 = ok7 and \
                    v7['restored_queries'] == v7['selected_injected']
        checks.append(check('S6', t, ok6, v6))
        checks.append(check("S7'", t, ok7, v7))
    m1 = rows.get('e1_missed_orig', {}).get('stats', {}).get('missed_gt')
    m2 = rows.get('e2_missed_boost', {}).get('stats', {}).get('missed_gt')
    for t, mg in (('e1_missed_orig', m1), ('e2_missed_boost', m2)):
        if mg is None or not nmd1:
            checks.append(check('S8', '%s missed_gt vs N_missed_D1' % t,
                                None, None))
        else:
            d = abs(mg - nmd1) / nmd1
            checks.append(check('S8', '%s missed_gt vs N_missed_D1' % t,
                                d <= 0.01, {'missed_gt': mg,
                                            'N_missed_D1': nmd1,
                                            'rel_diff': d}))
    if m1 and m2 is not None:
        d = abs(m1 - m2) / m1
        checks.append(check('S8', 'E1 vs E2 missed_gt', d <= 0.01,
                            {'e1': m1, 'e2': m2, 'rel_diff': d}))
    else:
        checks.append(check('S8', 'E1 vs E2 missed_gt', None, None))
    L.append('')
    L.append('Self-checks')
    for c in checks:
        val = json.dumps(c['value'], default=str)
        L.append('  %-4s %-40s %-8s %s' % (c['id'], c['desc'],
                                           c['result'], val))
    all_pass = all(c['result'] == 'PASS' for c in checks)
    L.append('  all self-checks PASS: %s' % all_pass)
    if e0 is not None:
        L.append('')
        L.append('Breakdown relative to E0 (oracle_breakdown.py --base '
                 'e0_base)')
        try:
            L += oracle_breakdown.breakdown_lines(root, 'e0_base')
        except (KeyError, ValueError, SystemExit) as e:
            L.append('breakdown unavailable: %r' % (e, ))
    res = {'rows': rows, 'coverage_orig_pp': rows.get(
        'e1_missed_orig', {}).get('dmAP_pp'),
        'coverage_boost_pp': rows.get('e2_missed_boost', {}).get('dmAP_pp'),
        'jitter_retention': retention, 'ref_dmap_all_pp': args.ref_dmap_all,
        'self_checks': checks, 'self_checks_all_pass': all_pass,
        'note': 'Q-rule verdicts are intentionally not computed.',
        'runtime': {'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'repo_version': read_version_line()}}
    write(root, 'exec2_summary', L, res)
    return 0


def main():
    p = argparse.ArgumentParser(description='exec doc #2 checks / summary')
    sub = p.add_subparsers(dest='cmd')
    a = sub.add_parser('s0p')
    a.add_argument('--config', required=True)
    a.add_argument('--ckpt', required=True)
    a.add_argument('--ref-json', required=True,
                   help="exec doc #1 step 3 base json (oracle_base.json)")
    a.add_argument('--out-dir', default='work_dirs/diag/exec2')
    b = sub.add_parser('step1-ok')
    b.add_argument('--root', default='work_dirs/diag/exec2')
    c = sub.add_parser('summary')
    c.add_argument('--root', default='work_dirs/diag/exec2')
    c.add_argument('--ref-dmap-all', type=float, default=1.875,
                   help='pre-registered denominator (exec doc #1 all group)')
    args = p.parse_args()
    if args.cmd == 's0p':
        return cmd_s0p(args)
    if args.cmd == 'step1-ok':
        return cmd_step1_ok(args)
    if args.cmd == 'summary':
        return cmd_summary(args)
    p.print_help()
    return 2


if __name__ == '__main__':
    sys.exit(main())
