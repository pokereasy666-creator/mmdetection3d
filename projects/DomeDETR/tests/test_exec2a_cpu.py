# -*- coding: utf-8 -*-
"""Dev-side CPU tests for exec doc #2a (tools/query_diag_exec2a); not
deployed. Run from the repo root (needs torch, numpy, mmengine):
  python projects/DomeDETR/tests/test_exec2a_cpu.py
The references are the real exec doc #2 step 2 / step 1 outputs archived in
projects/DomeDETR/results/exec02 (server runtime md5s, E0 metrics, B0).
"""
import hashlib
import itertools
import json
import os
import os.path as osp
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import test_query_diag_cpu as T  # noqa: E402

QD, MOCK, PY, REPO = T.QD, T.MOCK, T.PY, T.REPO
X2A = osp.join(REPO, 'tools', 'query_diag_exec2a')
CHECK = osp.join(X2A, 'exec2a_check.py')
RES2 = osp.join(REPO, 'projects/DomeDETR/results/exec02')
E0 = osp.join(RES2, 'step2_oracle', 'oracle_e0_base.json')
B0 = osp.join(RES2, 'step1_b0_decoder.json')
PRE = 'NuScenes metric/pred_instances_3d_NuScenes/'
THRS = ['0.5', '1.0', '2.0', '4.0']
case, sh = T.case, T.sh
CFG = CKPT = None


def md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


def run_clean(cmd, env, cwd):
    p = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True)
    return p.returncode, p.stdout


def fake_run(tag, d_thr, missed=20398, injected=20100, jitter=0,
             miss_thr=0.5, e0=None):
    """E3' json: every class AP at threshold t moved by d_thr[t] (absolute)."""
    r = json.load(open(e0 or E0))
    m = r['metrics']
    for k in list(m):
        for th in THRS:
            if k.startswith(PRE) and k.endswith('_AP_dist_' + th):
                m[k] = float(m[k]) + d_thr[th]
    m[PRE + 'mAP'] = float(m[PRE + 'mAP']) + sum(d_thr.values()) / 4
    r['args'].update({'tag': tag, 'oracle': 'loc', 'subset': 'missed',
                      'miss_thr': miss_thr, 'range': [0.0, 1000.0],
                      'score': 'orig', 'jitter_cells': jitter, 'seed': 0,
                      'no_class_range': False})
    r['stats'] = {'frames': 6019.0, 'injected_cells': float(injected),
                  'capped_cells': 0.0, 'restored_queries': float(injected),
                  'init_mismatch_rate': 0.0,
                  'selected_injected': float(injected),
                  'gt_in_range': 118538.0, 'missed_gt': float(missed)}
    return r


def put(root, r):
    tag = r['args']['tag']
    d = osp.join(root, 'oracle_%s' % tag)
    os.makedirs(d, exist_ok=True)
    json.dump(r, open(osp.join(d, 'oracle_%s.json' % tag), 'w'))


EX = {'0.5': 0.06, '1.0': 0.01, '2.0': 0.004, '4.0': 0.002}    # +1.9 pp
JI = {'0.5': -0.004, '1.0': 0.002, '2.0': 0.004, '4.0': 0.002}  # +0.1 pp


@case("2a summary: L, retention', r_0.5, checks on the real E0 / B0")
def test_summary(tmp):
    root = osp.join(tmp, 'sum', 'exec2a')
    put(root, fake_run('e3p_exact', EX))
    put(root, fake_run('e3p_jitter', JI, missed=20410, jitter=1))
    json.dump({'result': 'PASS'}, open(osp.join(root, 's0pp_check.json'),
                                       'w'))
    rc, log = sh([PY, CHECK, 'summary', '--root', root, '--e0', E0,
                  '--b0', B0])
    assert rc == 0, log
    s = json.load(open(osp.join(root, 'exec2a_summary.json')))
    d = s['derived']
    assert s['subset_ref'] == 20398, s['subset_ref']
    assert abs(d['exact_gain_pp'] - 1.9) < 1e-9, d
    assert abs(d['jitter_gain_pp'] - 0.1) < 1e-9, d
    assert abs(d['L_pp'] - 1.6) < 1e-9, d
    assert abs(d['retention_prime'] - 0.1 / 1.9) < 1e-9, d
    assert abs(d['r_0.5'] - (-0.4 / 6.0)) < 1e-9, d
    ch = s['rows']['e3p_exact']['share_pp']
    assert abs(ch['0.5'] - 1.5) < 1e-9 and abs(ch['4.0'] - 0.05) < 1e-9, ch
    assert s['self_checks_all_pass'] is True, s['self_checks']
    ids = [c['id'] for c in s['self_checks']]
    assert ids.count("S8'") == 3 and ids[0] == "S0''", ids
    txt = open(osp.join(root, 'exec2a_summary.txt')).read()
    assert '[no-NMS controls]' in txt
    ctrl = txt.split('[no-NMS controls]')[1].split('\n')[1:3]
    assert ctrl[0].split()[0] == 'pedestrian', ctrl
    assert ctrl[1].split()[0] == 'traffic_cone', ctrl
    assert 'TP errors and NDS' in txt and 'mAOE' in txt
    for w in ('C direction', 'closed', 'sufficient', 'premise'):
        assert w not in txt, w
    print('       exact %+.3f jitter %+.3f L %+.3f ret %+.3f r05 %+.3f'
          % (d['exact_gain_pp'], d['jitter_gain_pp'], d['L_pp'],
             d['retention_prime'], d['r_0.5']))


@case("2a summary: negatives (S8' 2%, ARGS, E0 file, exact < 0.5 pp)")
def test_summary_neg(tmp):
    seq = itertools.count()

    def run(exact, jitter, e0=E0):
        root = osp.join(tmp, 'neg_%d' % next(seq), 'exec2a')
        put(root, exact)
        put(root, jitter)
        rc, log = sh([PY, CHECK, 'summary', '--root', root, '--e0', e0,
                      '--b0', B0])
        s = json.load(open(osp.join(root, 'exec2a_summary.json')))
        return rc, {(c['id'], c['desc']): c['result']
                    for c in s['self_checks']}, s
    rc, r, _ = run(fake_run('e3p_exact', EX, missed=20808),
                   fake_run('e3p_jitter', JI, jitter=1))
    assert rc == 1 and r[("S8'", 'e3p_exact missed_gt vs B0 subset size')] \
        == 'FAIL', r
    rc, r, _ = run(fake_run('e3p_exact', EX, miss_thr=2.0),
                   fake_run('e3p_jitter', JI, jitter=0))
    assert rc == 1 and r[('ARGS', 'e3p_exact arguments as in exec doc #2a')] \
        == 'FAIL' and r[('ARGS', 'e3p_jitter arguments as in exec doc #2a')] \
        == 'FAIL', r
    e1 = osp.join(RES2, 'step2_oracle', 'oracle_e1_missed_orig.json')
    ji = fake_run('e3p_jitter', JI, jitter=1)
    rc, r, _ = run(fake_run('e3p_exact', EX), ji, e0=e1)
    assert rc == 1 and any(k[0] == 'E0' and v == 'FAIL'
                           for k, v in r.items()), r
    small = {'0.5': 0.004, '1.0': 0.0, '2.0': 0.0, '4.0': 0.0}  # +0.1 pp
    rc, r, s = run(fake_run('e3p_exact', small),
                   fake_run('e3p_jitter', JI, jitter=1))
    assert s['derived']['retention_prime'] is None, s['derived']
    assert abs(s['derived']['L_pp'] - (0.1 - 0.3)) < 1e-9


@case("2a check-run on the mock with the real oracle code: S8' == D1 subset")
def test_mock_runs(tmp):
    env = {'QD_MOCK_FRAMES': '12', 'QD_MOCK_SEED': '1'}
    rec = osp.join(tmp, 'mock_recall')
    rc, log = sh([PY, MOCK, 'recall', QD, rec, CFG, CKPT], env=env)
    assert rc == 0, log[-2000:]
    Z = np.load(osp.join(rec, 'recall_per_gt.npz'))
    j05 = list(Z['thr']).index(0.5)
    n_sub = int((Z['rank_ca'][:, j05] >= int(Z['P'])).sum())
    b0d = osp.join(tmp, 'mock_b0')
    sh([PY, osp.join(QD, 'diag_b0_decoder.py'), '--npz',
        osp.join(rec, 'recall_per_gt.npz'), '--out-dir', b0d])
    b0 = osp.join(b0d, 'b0_decoder.json')
    out = {}
    for tag, extra in (('e3p_exact', []),
                       ('e3p_jitter', ['--jitter-cells', '1', '--seed',
                                       '0'])):
        wd = osp.join(tmp, 'mock_x2a', 'oracle_' + tag)
        rc, log = sh([PY, MOCK, 'oracle', QD, wd, CFG, CKPT, tag,
                      '--oracle', 'loc', '--subset', 'missed', '--miss-thr',
                      '0.5', '--range', '0', '1000', '--score', 'orig'] +
                     extra, env=env)
        assert rc == 0, log[-2000:]
        j = osp.join(wd, 'oracle_%s.json' % tag)
        rc, log = sh([PY, CHECK, 'check-run', '--json', j, '--b0', b0])
        assert rc == 0, log
        out[tag] = json.load(open(j))['stats']
    assert out['e3p_exact']['missed_gt'] == n_sub == \
        out['e3p_jitter']['missed_gt'], (out, n_sub)
    assert out['e3p_exact']['selected_injected'] == \
        out['e3p_exact']['restored_queries'] > 0
    print('       subset (rank_ca@0.5 >= P) = %d; injected exact %d jitter %d'
          % (n_sub, out['e3p_exact']['injected_cells'],
             out['e3p_jitter']['injected_cells']))


def ref_dir(tmp, name, cfg_opts):
    """Copies of the real exec doc #2 step 2 jsons, CONFIG / CKPT md5 set to
    the local fake files (diag_files_md5 stays the real server record)."""
    d = osp.join(tmp, name, 'work_dirs', 'diag', 'exec2')
    for f in os.listdir(osp.join(RES2, 'step2_oracle')):
        if not f.startswith('oracle_'):
            continue
        r = json.load(open(osp.join(RES2, 'step2_oracle', f)))
        r['runtime']['config_md5'] = md5(CFG)
        r['runtime']['checkpoint_md5'] = md5(CKPT)
        r['runtime']['cfg_opts_env'] = cfg_opts
        sub = osp.join(d, f[:-5])
        os.makedirs(sub)
        json.dump(r, open(osp.join(sub, f), 'w'))
    shutil.copy(B0, osp.join(d, 'b0_decoder.json'))
    return d


@case("2a S0'': repo tools/query_diag == server step 2 record; negatives")
def test_s0pp(tmp):
    ref = ref_dir(tmp, 's0pp', 'X=1')
    out = osp.join(tmp, 's0pp_out')
    base = [PY, CHECK, 's0pp', '--ref-dir', ref, '--config', CFG, '--ckpt',
            CKPT, '--out-dir', out]
    rc, log = sh(base, env={'CFG_OPTS': 'X=1'})
    assert rc == 0 and "S0''  PASS" in log and 'all identical' in log, log
    rc, log = sh(base, env={'CFG_OPTS': 'X=2'})
    assert rc == 1 and 'CFG_OPTS differs' in log, log
    qd2 = osp.join(tmp, 'qd_copy')
    shutil.copytree(QD, qd2, ignore=shutil.ignore_patterns('__pycache__'))
    with open(osp.join(qd2, 'diag_common.py'), 'a') as f:
        f.write('\n')
    open(osp.join(qd2, 'extra_tool.py'), 'w').write('x = 1\n')
    rc, log = sh(base + ['--qd', qd2], env={'CFG_OPTS': 'X=1'})
    assert rc == 1 and 'diag_common.py CHANGED' in log and \
        'extra_tool.py NEW' in log, log
    r = osp.join(ref, 'oracle_e3_jitter', 'oracle_e3_jitter.json')
    j = json.load(open(r))
    j['runtime']['diag_files_md5']['run_diag.sh'] = '0' * 32
    json.dump(j, open(r, 'w'))
    rc, log = sh(base, env={'CFG_OPTS': 'X=1'})
    assert rc == 1 and 'DISAGREE' in log, log


PYSTUB = r'''#!/usr/bin/env bash
echo "python $*" >> "$STUB_LOG"
case "$*" in
  *exec2a_check.py*) exec "$REAL_PY" "$@" ;;
esac
if [[ "$*" == *diag_oracle_query.py* ]]; then
  args=("$@"); wd=""; tag=""
  for ((i=0; i<${#args[@]}; i++)); do
    case "${args[$i]}" in
      --work-dir) wd=${args[$((i+1))]} ;;
      --tag) tag=${args[$((i+1))]} ;;
    esac
  done
  mkdir -p "$wd"
  cp "$STUB_DIR/oracle_$tag.json" "$wd/oracle_$tag.json"
fi
exit 0
'''


@case('2a run_exec2a.sh: order, verbatim args, gating, C5, logs')
def test_runner(tmp):
    b = osp.join(tmp, 'x2a_bin')
    os.makedirs(b)
    for name, body in (('nvidia-smi', T.NVSMI), ('python', PYSTUB)):
        open(osp.join(b, name), 'w').write(body)
        os.chmod(osp.join(b, name), 0o755)
    good = osp.join(tmp, 'stub_good')
    os.makedirs(good)
    for r in (fake_run('e3p_exact', EX),
              fake_run('e3p_jitter', JI, jitter=1)):
        json.dump(r, open(osp.join(good, 'oracle_%s.json'
                                   % r['args']['tag']), 'w'))
    bad = osp.join(tmp, 'stub_bad')
    shutil.copytree(good, bad)
    json.dump(fake_run('e3p_exact', EX, missed=21000),
              open(osp.join(bad, 'oracle_e3p_exact.json'), 'w'))
    slog = osp.join(tmp, 'x2a_stub.log')
    seq = itertools.count()

    def run(step, **env):
        ws = osp.join(tmp, 'x2a_ws')
        shutil.rmtree(ws, ignore_errors=True)
        T.make_workspace(tmp, 'x2a_ws')
        shutil.rmtree(osp.join(ws, 'tools', 'query_diag'))
        for src in (QD, X2A):
            shutil.copytree(src, osp.join(ws, 'tools', osp.basename(src)),
                            ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(ref_dir(tmp, 'rd_%d' % next(seq), 'X=1'),
                        osp.join(ws, 'work_dirs', 'diag', 'exec2'))
        if osp.exists(slog):
            os.remove(slog)
        e = {'PATH': b + os.pathsep + os.environ['PATH'], 'STUB_LOG': slog,
             'REAL_PY': PY, 'STUB_DIR': good, 'CONFIG': CFG, 'CKPT': CKPT,
             'CFG_OPTS': 'X=1', 'CUDA_VISIBLE_DEVICES': '2'}
        e.update(env)
        rc, txt = run_clean(['bash', 'tools/query_diag_exec2a/'
                             'run_exec2a.sh', step], e, ws)
        calls = open(slog).read().splitlines() if osp.exists(slog) else []
        return rc, txt, calls, ws

    def kinds(calls):
        out = []
        for c in calls:
            if 'diag_oracle_query.py' in c:
                out.append('oracle:' + c.split('--tag ')[1].split()[0])
            elif 'exec2a_check.py' in c:
                out.append(c.split('exec2a_check.py ')[1].split()[0])
        return out

    rc, txt, calls, ws = run('run')
    assert rc == 0, txt
    assert kinds(calls) == ['s0pp', 'oracle:e3p_exact', 'check-run',
                            'oracle:e3p_jitter', 'summary'], kinds(calls)
    oc = [c for c in calls if 'diag_oracle_query.py' in c]
    assert ('--oracle loc --subset missed --miss-thr 0.5 --range 0 1000 '
            '--score orig --cfg-options X=1') in oc[0], oc[0]
    assert ('--oracle loc --subset missed --miss-thr 0.5 --range 0 1000 '
            '--score orig --jitter-cells 1 --seed 0 --cfg-options X=1') \
        in oc[1], oc[1]
    assert all('/tools/query_diag/diag_oracle_query.py' in c for c in oc)
    assert all('--work-dir work_dirs/diag/exec2a/oracle_' in c for c in oc)
    assert txt.count('[gpu-check] GPU 2 idle') == 2, txt
    x = osp.join(ws, 'work_dirs', 'diag', 'exec2a')
    assert sorted(os.listdir(x)) == ['exec2a_summary.json',
                                     'exec2a_summary.txt', 'oracle_e3p_exact',
                                     'oracle_e3p_jitter', 's0pp_check.json',
                                     's0pp_check.txt'], os.listdir(x)
    logs = os.listdir(osp.join(ws, 'work_dirs', 'diag', 'logs'))
    assert len(logs) == 1 and re.match(r'exec2a-run_\d{8}_\d{6}\.log$',
                                       logs[0]), logs
    assert 'exec2a-run_' in open(osp.join(x, 'exec2a_summary.txt')).read()
    rc, txt, calls, _ = run('run', CFG_OPTS='X=2')
    assert rc == 4 and kinds(calls) == ['s0pp'], (kinds(calls), txt)
    rc, txt, calls, ws = run('run', STUB_DIR=bad)
    assert rc == 4 and 'E3\'-jitter was NOT run' in txt, txt
    assert kinds(calls) == ['s0pp', 'oracle:e3p_exact', 'check-run',
                            'summary'], kinds(calls)
    rc, txt, calls, _ = run('run', CUDA_VISIBLE_DEVICES='1,2')
    assert rc == 2 and 'exactly 1' in txt and kinds(calls) == ['s0pp'], txt
    rc, txt, calls, _ = run('run', STUB_MEM_2='2048')
    assert rc == 3 and kinds(calls) == ['s0pp'], txt
    rc, txt, calls, _ = run('nonsense')
    assert rc == 1 and 'usage' in txt and not calls, txt
    rc, txt, calls, ws = run('run', PROTECTED_DIRS=osp.join(tmp, 'x2a_ws'))
    assert rc == 2 and 'protected' in txt and not calls, txt


def fn_body(text, name):
    m = re.search(r'^%s\(\) \{\n(.*?)^\}\n' % re.escape(name), text,
                  re.S | re.M)
    return m.group(1) if m else None


@case('2a run_exec2a.sh guards are verbatim copies of run_diag.sh')
def test_verbatim(tmp):
    a = open(osp.join(QD, 'run_diag.sh')).read()
    b = open(osp.join(X2A, 'run_exec2a.sh')).read()
    for name in ('inside_protected', 'need_repo_root', 'need_config',
                 'need_ckpt', 'check_gpus'):
        x, y = fn_body(a, name), fn_body(b, name)
        assert x and x == y, name
    for line in ('MEM_LIMIT_MIB=1024', 'export CUDA_DEVICE_ORDER=PCI_BUS_ID',
                 'PROTECTED_DIRS=${PROTECTED_DIRS:-/212022085500129/'
                 'bevfusion-concat/mmdetection3d-claude-jolly-wozniak-'
                 'c4YMQ}', 'read -r -a _opts <<< "$CFG_OPTS"',
                 'CFG_ARGS=(--cfg-options "${_opts[@]}")'):
        assert line in a and line in b, line
    assert 'torchrun' not in b and 'check_gpus 1' in b


def main():
    global CFG, CKPT
    tmp = tempfile.mkdtemp(prefix='qd_exec2a_')
    CFG = osp.join(tmp, 'fake_config.py')
    CKPT = osp.join(tmp, 'fake_ckpt.pth')
    open(CFG, 'w').write('# fake\n')
    open(CKPT, 'wb').write(os.urandom(1024))
    test_summary(tmp)
    test_summary_neg(tmp)
    test_mock_runs(tmp)
    test_s0pp(tmp)
    test_runner(tmp)
    test_verbatim(tmp)
    n_fail = sum(1 for r in T.RESULTS if r[1] != 'PASS')
    print('\n%d tests, %d failed' % (len(T.RESULTS), n_fail))
    shutil.rmtree(tmp, ignore_errors=True)
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
