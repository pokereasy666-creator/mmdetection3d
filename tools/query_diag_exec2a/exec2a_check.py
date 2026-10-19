#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Checks and summary for exec doc #2a (E3': the localization gain measured
on GT without a same-class initial query within 0.5 m).

Lives outside tools/query_diag on purpose: S0'' requires every file there to
be byte-identical to exec doc #2 step 2. The two E3' runs use the unchanged
tools/query_diag/diag_oracle_query.py with existing flags only.

s0pp       S0'': tools/query_diag (every .py/.sh/.json), CONFIG and CKPT md5
           and CFG_OPTS equal to the runtime recorded by the exec doc #2
           step 2 runs (work_dirs/diag/exec2/oracle_*/oracle_*.json); the
           files of this directory pass the sync gate.
check-run  S6, S7', S8' (vs the B0 subset size) and the argument check for one
           E3' json; run_exec2a.sh stops before E3'-jitter if E3'-exact fails.
summary    dmAP / dNDS vs the exec doc #2 E0, L = exact_gain - 0.300,
           retention' (only when exact_gain >= 0.5 pp), r_0.5, per-threshold
           and per-class breakdown (pedestrian / traffic_cone listed apart),
           mATE / mAOE, S0'' / S6 / S7' / S8', runtime. Q4 / Q3' verdicts are
           intentionally NOT computed.
Pure json / stdlib (+ tools/query_diag/oracle_breakdown.py); no torch.
"""
import argparse
import ast
import glob
import hashlib
import json
import os
import os.path as osp
import shutil
import subprocess
import sys
import tempfile
import time

HERE = osp.dirname(osp.abspath(__file__))
QD_DEFAULT = osp.join(osp.dirname(HERE), 'query_diag')

PRE = 'NuScenes metric/pred_instances_3d_NuScenes/'
CLASSES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
           'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
NO_NMS = ('pedestrian', 'traffic_cone')   # no 3x3 local max in the head
THRS = ['0.5', '1.0', '2.0', '4.0']
TAGS = ('e3p_exact', 'e3p_jitter')
# exec doc #2a sec. 2: the only two argument sets allowed
EXPECT_ARGS = {
    'e3p_exact': {'oracle': 'loc', 'subset': 'missed', 'miss_thr': 0.5,
                  'range': [0.0, 1000.0], 'score': 'orig', 'jitter_cells': 0,
                  'seed': 0, 'no_class_range': False},
    'e3p_jitter': {'oracle': 'loc', 'subset': 'missed', 'miss_thr': 0.5,
                   'range': [0.0, 1000.0], 'score': 'orig', 'jitter_cells': 1,
                   'seed': 0, 'no_class_range': False},
}


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
    print('[exec2a] saved %s/%s.{txt,json}' % (out_dir, stem))


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
    return ''


def check(cid, desc, value, ok):
    return {'id': cid, 'desc': desc, 'value': value,
            'result': 'PASS' if ok else 'FAIL'}


def pick(m, key):
    return float(m[PRE + key])


def cm_ap(m, th):
    """class-mean AP at one match threshold."""
    return sum(pick(m, '%s_AP_dist_%s' % (c, th)) for c in CLASSES) / \
        len(CLASSES)


# ---------------------------------------------------------------------------
# S0''
# ---------------------------------------------------------------------------

def cmd_s0pp(args):
    L = ["S0'' consistency with exec doc #2 step 2 (%s)" % args.ref_dir]
    fail = []
    refs = {}
    for p in sorted(glob.glob(osp.join(args.ref_dir, 'oracle_*',
                                       'oracle_*.json'))):
        rt = load(p).get('runtime', {})
        refs[osp.basename(p)] = {
            'diag_files_md5': rt.get('diag_files_md5'),
            'config_md5': rt.get('config_md5'),
            'checkpoint_md5': rt.get('checkpoint_md5'),
            'cfg_opts_env': rt.get('cfg_opts_env')}
    if not refs:
        fail.append('no exec doc #2 step 2 oracle json under %s'
                    % args.ref_dir)
        ref = {}
    else:
        names = sorted(refs)
        ref = refs[names[0]]
        same = all(refs[n] == ref for n in names)
        L.append('  reference runs: %s (runtime records %s)'
                 % (', '.join(names), 'agree' if same else 'DISAGREE'))
        if not same:
            fail.append('the step 2 runs do not share one runtime record')
    cm, km = file_md5(args.config), file_md5(args.ckpt)
    opts = os.environ.get('CFG_OPTS', '')
    for name, now, want in (('CONFIG md5', cm, ref.get('config_md5')),
                            ('CKPT md5', km, ref.get('checkpoint_md5')),
                            ('CFG_OPTS', opts, ref.get('cfg_opts_env'))):
        ok = now == want
        L.append('  %-10s %s  (#2: %s)  %s'
                 % (name, now, want, 'OK' if ok else 'DIFF'))
        if not ok:
            fail.append('%s differs from exec doc #2 step 2' % name)
    ref_md5 = ref.get('diag_files_md5') or {}
    cur = {f: file_md5(osp.join(args.qd, f))
           for f in sorted(os.listdir(args.qd))
           if f.endswith(('.py', '.sh', '.json'))}
    diffs = []
    for f in sorted(set(ref_md5) | set(cur)):
        if ref_md5.get(f) != cur.get(f):
            st = 'NEW' if f not in ref_md5 else ('GONE' if f not in cur
                                                 else 'CHANGED')
            diffs.append({'file': f, 'status': st, 'md5_2': ref_md5.get(f),
                          'md5_now': cur.get(f)})
            fail.append('tools/query_diag/%s %s since exec doc #2 step 2'
                        % (f, st))
    L.append('  tools/query_diag: %d files, %s'
             % (len(cur), 'all identical to exec doc #2 step 2' if not diffs
                else '%d differ' % len(diffs)))
    for d in diffs:
        L.append('    %-26s %-8s %-8s -> %-8s'
                 % (d['file'], d['status'], (d['md5_2'] or '-')[:8],
                    (d['md5_now'] or '-')[:8]))
    own = {f: file_md5(osp.join(HERE, f)) for f in sorted(os.listdir(HERE))
           if f.endswith(('.py', '.sh'))}
    gates = {f: gate_one(osp.join(HERE, f)) for f in own}
    bad = {f: e for f, e in gates.items() if e}
    L.append('  tools/query_diag_exec2a: %s; sync gate %s'
             % (', '.join('%s %s' % (f, m[:8]) for f, m in own.items()),
                'all OK' if not bad else bad))
    for f, e in bad.items():
        fail.append('sync gate: %s %s' % (f, e))
    for x in fail:
        L.append('  FAIL ' + x)
    res = {'check': "S0''", 'result': 'FAIL' if fail else 'PASS',
           'fail': fail, 'config_md5': cm, 'checkpoint_md5': km,
           'cfg_opts_env': opts, 'query_diag_diffs': diffs,
           'exec2a_files_md5': own, 'gate': gates,
           'runtime': {'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                       'repo_version': read_version_line()}}
    L.append("S0''  %s" % res['result'])
    write(args.out_dir, 's0pp_check', L, res)
    return 0 if not fail else 1


# ---------------------------------------------------------------------------
# per-run checks
# ---------------------------------------------------------------------------

def subset_ref(b0):
    """S8' reference: N_gt x (1 - class-aware init recall @0.5 m, K=200)."""
    n = int(b0['num_gt'])
    r = float(b0['B0_1']['all']['thr']['0.5']['init'])
    return int(round(n * (1.0 - r))), n, r


def run_checks(tag, r, ref_n):
    st, a = r['stats'], r['args']
    out = []
    want = EXPECT_ARGS[tag]
    got = {k: a.get(k) for k in want}
    if got.get('range') is not None:
        got['range'] = [float(x) for x in got['range']]
    out.append(check('ARGS', '%s arguments as in exec doc #2a' % tag, got,
                     got == want))
    out.append(check('S6', tag, {'init_mismatch_rate':
                                 st.get('init_mismatch_rate'),
                                 'capped_cells': st.get('capped_cells')},
                     st.get('init_mismatch_rate', 1) <= 0.01 and
                     st.get('capped_cells', 1) == 0))
    sel, inj = st.get('selected_injected'), st.get('injected_cells')
    rest = st.get('restored_queries')
    out.append(check("S7'", tag, {'selected_injected': sel,
                                  'injected_cells': inj,
                                  'restored_queries': rest},
                     sel is not None and inj is not None and inj > 0 and
                     sel >= 0.99 * inj and rest == sel))
    mg = st.get('missed_gt')
    rel = abs(mg - ref_n) / ref_n if mg is not None else None
    out.append(check("S8'", '%s missed_gt vs B0 subset size' % tag,
                     {'missed_gt': mg, 'subset_ref': ref_n, 'rel_diff': rel},
                     rel is not None and rel <= 0.01))
    return out


def cmd_check_run(args):
    r = load(args.json)
    tag = r['args']['tag']
    ref_n, _, _ = subset_ref(load(args.b0))
    if tag not in EXPECT_ARGS:
        print('[exec2a] unexpected tag %r' % tag)
        return 1
    cs = run_checks(tag, r, ref_n)
    for c in cs:
        print('[exec2a] %-5s %-40s %s  %s'
              % (c['id'], c['desc'], c['result'], json.dumps(c['value'])))
    return 0 if all(c['result'] == 'PASS' for c in cs) else 1


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def breakdown(qd, e0_path, run_paths):
    """oracle_breakdown.py output for E0 + the E3' runs (unchanged code)."""
    sys.path.insert(0, qd)
    import oracle_breakdown
    tmp = tempfile.mkdtemp(prefix='exec2a_bd_')
    try:
        for p in [e0_path] + run_paths:
            d = osp.join(tmp, osp.basename(osp.dirname(p)))
            os.makedirs(d)
            shutil.copy(p, d)
        return oracle_breakdown.breakdown_lines(tmp, 'e0_base')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def cmd_summary(args):
    e0 = load(args.e0)
    b0 = load(args.b0)
    ref_n, n_gt, r05 = subset_ref(b0)
    paths = {t: osp.join(args.root, 'oracle_%s' % t, 'oracle_%s.json' % t)
             for t in TAGS}
    runs = {t: load(p) for t, p in paths.items() if osp.isfile(p)}
    m0 = e0['metrics']
    L = ["Exec doc #2a summary (E3'; dmAP / dNDS in pp vs exec doc #2 E0; "
         "no Q4 / Q3' verdicts)"]
    checks = []
    e0_ok = (e0['args'].get('tag') == 'e0_base' and
             '%.6f' % pick(m0, 'mAP') == args.e0_map and
             '%.6f' % pick(m0, 'NDS') == args.e0_nds)
    checks.append(check('E0', 'reference = exec doc #2 E0 (%s / %s)'
                        % (args.e0_map, args.e0_nds),
                        {'mAP': pick(m0, 'mAP'), 'NDS': pick(m0, 'NDS')},
                        e0_ok))
    L.append('E0 (%s): mAP %.6f  NDS %.6f' % (args.e0, pick(m0, 'mAP'),
                                              pick(m0, 'NDS')))
    L.append("S8' subset reference: N_gt %d x (1 - init@0.5m %.5f) = %d"
             % (n_gt, r05, ref_n))
    L.append('')
    L.append('%-11s %8s %8s %8s %8s %9s %9s %9s %9s %8s %8s'
             % ('tag', 'mAP', 'dmAP', 'NDS', 'dNDS', 'injected', 'selected',
                'restored', 'missed', 'mATE', 'mAOE'))
    L.append('%-11s %8.4f %8s %8.4f %8s %39s %8.4f %8.4f'
             % ('e0_base', pick(m0, 'mAP'), '', pick(m0, 'NDS'), '', '',
                pick(m0, 'mATE'), pick(m0, 'mAOE')))
    rows = {}
    for t in TAGS:
        r = runs.get(t)
        if r is None:
            L.append('%-11s MISSING (%s)' % (t, paths[t]))
            continue
        m, st = r['metrics'], r['stats']
        dm = 100 * (pick(m, 'mAP') - pick(m0, 'mAP'))
        dn = 100 * (pick(m, 'NDS') - pick(m0, 'NDS'))
        share = {th: 100 * (cm_ap(m, th) - cm_ap(m0, th)) / len(THRS)
                 for th in THRS}
        rows[t] = {'mAP': pick(m, 'mAP'), 'NDS': pick(m, 'NDS'),
                   'dmAP_pp': dm, 'dNDS_pp': dn, 'share_pp': share,
                   'mATE': pick(m, 'mATE'), 'mAOE': pick(m, 'mAOE'),
                   'stats': st}
        L.append('%-11s %8.4f %+8.3f %8.4f %+8.3f %9d %9d %9d %9d %8.4f %8.4f'
                 % (t, pick(m, 'mAP'), dm, pick(m, 'NDS'), dn,
                    st.get('injected_cells', 0),
                    st.get('selected_injected', 0),
                    st.get('restored_queries', 0), st.get('missed_gt', 0),
                    pick(m, 'mATE'), pick(m, 'mAOE')))
        checks += run_checks(t, r, ref_n)
    if len(rows) == 2:
        a = rows['e3p_exact']['stats'].get('missed_gt')
        b = rows['e3p_jitter']['stats'].get('missed_gt')
        rel = abs(a - b) / max(a, b) if a and b else None
        checks.append(check("S8'", 'e3p_exact vs e3p_jitter missed_gt',
                            {'exact': a, 'jitter': b, 'rel_diff': rel},
                            rel is not None and rel <= 0.01))
    L.append('')
    L.append('dmAP contribution by match threshold (class-mean dAP / 4), pp')
    for t in TAGS:
        if t in rows:
            s = rows[t]['share_pp']
            L.append('  %-11s ' % t + '  '.join('@%sm %+7.3f' % (th, s[th])
                                                for th in THRS) +
                     '  | dmAP %+7.3f' % rows[t]['dmAP_pp'])

    derived = {'coverage_orig_registered_pp': args.coverage_orig}
    if 'e3p_exact' in rows:
        ex = rows['e3p_exact']['dmAP_pp']
        derived['exact_gain_pp'] = ex
        derived['L_pp'] = ex - args.coverage_orig
    if 'e3p_jitter' in rows:
        derived['jitter_gain_pp'] = rows['e3p_jitter']['dmAP_pp']
    if len(rows) == 2:
        ex, ji = derived['exact_gain_pp'], derived['jitter_gain_pp']
        derived['retention_prime'] = ji / ex if ex >= args.min_exact else None
        s_ex = rows['e3p_exact']['share_pp']['0.5']
        s_ji = rows['e3p_jitter']['share_pp']['0.5']
        derived['r_0.5'] = s_ji / s_ex if s_ex > 0 else None
    L.append('')

    def fmt(k, f):
        v = derived.get(k)
        return f % v if isinstance(v, float) else 'n/a'
    L.append("exact_gain  = dmAP(E3'-exact)            = %s pp"
             % fmt('exact_gain_pp', '%+.3f'))
    L.append("jitter_gain = dmAP(E3'-jitter)           = %s pp"
             % fmt('jitter_gain_pp', '%+.3f'))
    L.append('L           = exact_gain - %.3f           = %s pp'
             % (args.coverage_orig, fmt('L_pp', '%+.3f')))
    L.append("retention'  = jitter_gain / exact_gain   = %s%s"
             % (fmt('retention_prime', '%+.3f'),
                '' if derived.get('retention_prime') is not None or
                'exact_gain_pp' not in derived else
                '  (not computed: exact_gain < %.1f pp)' % args.min_exact))
    L.append("r_0.5       = @0.5m share jitter / exact = %s  (descriptive)"
             % fmt('r_0.5', '%+.3f'))

    L.append('')
    L.append('Per-class dAP vs E0, pp, @0.5 / 1 / 2 / 4 m (classes without '
             '3x3 local max listed apart)')
    per_class = {}
    for group, cls in (('local-max classes', [c for c in CLASSES
                                              if c not in NO_NMS]),
                       ('no-NMS controls', list(NO_NMS))):
        L.append('  [%s]' % group)
        for c in cls:
            row = {}
            cells = []
            for t in TAGS:
                if t not in runs:
                    continue
                d = [100 * (float(runs[t]['metrics'][PRE + '%s_AP_dist_%s'
                                                     % (c, th)]) -
                            float(m0[PRE + '%s_AP_dist_%s' % (c, th)]))
                     for th in THRS]
                row[t] = d
                cells.append('%s ' % t.replace('e3p_', '') +
                             ' '.join('%+6.2f' % x for x in d))
            per_class[c] = row
            L.append('  %-20s %s' % (c, '   '.join(cells)))

    L.append('')
    L.append('Breakdown relative to E0 (tools/query_diag/oracle_breakdown.py, '
             'unchanged)')
    try:
        L += breakdown(args.qd, args.e0, [paths[t] for t in TAGS
                                          if t in runs])
    except (SystemExit, Exception) as e:  # noqa: B902
        L.append('breakdown unavailable: %r' % (e, ))

    s0 = osp.join(args.root, 's0pp_check.json')
    s0_res = load(s0)['result'] if osp.isfile(s0) else 'MISSING'
    checks.insert(0, check("S0''", 'consistency with exec doc #2 step 2',
                           s0_res, s0_res == 'PASS'))
    L.append('')
    L.append('Self-checks')
    for c in checks:
        val = json.dumps(c['value'], default=str)
        L.append('  %-5s %-40s %-5s %s' % (c['id'], c['desc'], c['result'],
                                           val))
    all_pass = all(c['result'] == 'PASS' for c in checks)
    L.append('  all self-checks PASS: %s' % all_pass)

    L.append('')
    L.append('Runtime')
    rts = {}
    for t in TAGS:
        if t in runs:
            rt = runs[t].get('runtime', {})
            rts[t] = {k: rt.get(k) for k in (
                'elapsed_s', 'gpu_name', 'cuda_visible_devices', 'world_size',
                'fp16', 'repo_version', 'peak_mem_allocated_mib')}
            L.append('  %-11s %s' % (t, json.dumps(rts[t], default=str)))
    logs = sorted(glob.glob(osp.join(osp.dirname(osp.abspath(args.root)),
                                     'logs', 'exec2a-run_*.log')))
    L.append('  logs: %s' % (', '.join(logs) if logs else 'none found'))
    L.append('')
    L.append("Q4 / Q3' verdicts are intentionally not computed (exec doc #2a "
             'sec. 5, review side).')
    res = {'e0': args.e0, 'b0': args.b0, 'subset_ref': ref_n,
           'rows': rows, 'derived': derived, 'per_class_dAP_pp': per_class,
           'self_checks': checks, 'self_checks_all_pass': all_pass,
           'runtime': rts, 'logs': logs,
           'summary_time': time.strftime('%Y-%m-%d %H:%M:%S'),
           'repo_version': read_version_line()}
    write(args.root, 'exec2a_summary', L, res)
    return 0 if all_pass else 1


def main():
    p = argparse.ArgumentParser(description='exec doc #2a checks / summary')
    sub = p.add_subparsers(dest='cmd')
    s = sub.add_parser('s0pp')
    s.add_argument('--qd', default=QD_DEFAULT)
    s.add_argument('--ref-dir', default='work_dirs/diag/exec2')
    s.add_argument('--config', required=True)
    s.add_argument('--ckpt', required=True)
    s.add_argument('--out-dir', default='work_dirs/diag/exec2a')
    c = sub.add_parser('check-run')
    c.add_argument('--json', required=True)
    c.add_argument('--b0', default='work_dirs/diag/exec2/b0_decoder.json')
    m = sub.add_parser('summary')
    m.add_argument('--root', default='work_dirs/diag/exec2a')
    m.add_argument('--qd', default=QD_DEFAULT)
    m.add_argument('--e0', default='work_dirs/diag/exec2/oracle_e0_base/'
                   'oracle_e0_base.json')
    m.add_argument('--b0', default='work_dirs/diag/exec2/b0_decoder.json')
    m.add_argument('--e0-map', default='0.666180',
                   help='exec doc #2a sec. 1 E0 mAP (6 decimals)')
    m.add_argument('--e0-nds', default='0.706324',
                   help='exec doc #2a sec. 1 E0 NDS (6 decimals)')
    m.add_argument('--coverage-orig', type=float, default=0.300,
                   help='registered coverage_orig of exec doc #2 E1, pp')
    m.add_argument('--min-exact', type=float, default=0.5,
                   help="retention' only when exact_gain >= this, pp")
    args = p.parse_args()
    if args.cmd == 's0pp':
        return cmd_s0pp(args)
    if args.cmd == 'check-run':
        return cmd_check_run(args)
    if args.cmd == 'summary':
        return cmd_summary(args)
    p.print_help()
    return 2


if __name__ == '__main__':
    sys.exit(main())
