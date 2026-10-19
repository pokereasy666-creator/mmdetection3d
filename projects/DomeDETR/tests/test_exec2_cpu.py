# -*- coding: utf-8 -*-
"""Dev-side CPU tests for the exec doc #2 changes (M1-M5); not deployed.

Run from the repo root (needs torch, numpy, mmengine, git):
  python projects/DomeDETR/tests/test_exec2_cpu.py
The exec doc #1 implementation of diag_oracle_query.py / diag_common.py is
taken from commit de1bbcd (md5 equal to the values recorded in the exec doc
#1 server runs) to prove that '--subset all' behaves exactly as before.
"""
import copy
import hashlib
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
EXEC1_COMMIT = 'de1bbcd'
REF1 = osp.join(REPO, 'projects/DomeDETR/results/exec01/step3_oracle/'
                'oracle_base.json')
case, sh = T.case, T.sh


def md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


def oracle(tmp, impl, tag, extra, env=None):
    wd = osp.join(tmp, 'o_' + osp.basename(impl) + '_' + tag)
    rc, log = sh([PY, MOCK, 'oracle', impl, wd, CFG, CKPT, tag] + extra,
                 env=env)
    assert rc == 0, log[-3000:]
    r = json.load(open(osp.join(wd, 'oracle_%s.json' % tag)))
    h = json.load(open(osp.join(wd, 'harness_checks_%s.json' % tag)))
    return r, h


def extract_exec1(tmp):
    d = osp.join(tmp, 'impl_exec1')
    os.makedirs(d)
    for f in ('diag_oracle_query.py', 'diag_common.py'):
        src = subprocess.check_output(
            ['git', 'show', '%s:tools/query_diag/%s' % (EXEC1_COMMIT, f)],
            cwd=REPO)
        open(osp.join(d, f), 'wb').write(src)
    ref = json.load(open(REF1))['runtime']['diag_files_md5']
    for f in ('diag_oracle_query.py', 'diag_common.py'):
        assert md5(osp.join(d, f)) == ref[f], f
    return d


LOC = ['--oracle', 'loc', '--range', '0', '1000']
JIT = LOC + ['--jitter-cells', '1', '--seed', '0']


@case("exec2 M1: '--subset all' identical to the exec doc #1 code")
def test_subset_all_identical(tmp):
    old = extract_exec1(tmp)
    for name, extra in (('loc', LOC), ('jitter', JIT),
                        ('base', ['--oracle', 'none'])):
        r_old, _ = oracle(tmp, old, name, extra)
        r_new, _ = oracle(tmp, QD, name, extra)
        assert r_new['metrics'] == r_old['metrics'], name
        for k, v in r_old['stats'].items():
            assert r_new['stats'][k] == v, (name, k, v, r_new['stats'][k])
        assert r_new['args']['subset'] == 'all'
        if name != 'base':
            st = r_new['stats']
            assert st['selected_injected'] == st['injected_cells'] > 0, st
            assert st['restored_queries'] == st['selected_injected'], st
            assert st['gt_in_range'] > st['missed_gt'] > 0, st
        print('       %-6s metrics and stats identical to #1' % name)


@case('exec2 M1/M2: missed subset == D1 definition; orig vs boost')
def test_missed(tmp):
    env = {'QD_MOCK_FRAMES': '12', 'QD_MOCK_SEED': '1'}
    out = osp.join(tmp, 'recall_same_frames')
    rc, log = sh([PY, MOCK, 'recall', QD, out, CFG, CKPT], env=env)
    assert rc == 0, log[-2000:]
    Z = np.load(osp.join(out, 'recall_per_gt.npz'))
    ti = list(Z['thr']).index(2.0)
    n_missed = int((Z['rank_ca'][:, ti] >= int(Z['P'])).sum())
    e1, h1 = oracle(tmp, QD, 'e1', LOC + ['--subset', 'missed', '--score',
                                          'orig'], env)
    e2, h2 = oracle(tmp, QD, 'e2', LOC + ['--subset', 'missed', '--score',
                                          'boost'], env)
    ea, _ = oracle(tmp, QD, 'eall', LOC, env)
    s1, s2, sa = e1['stats'], e2['stats'], ea['stats']
    assert s1['missed_gt'] == s2['missed_gt'] == sa['missed_gt'] == n_missed
    assert s1['gt_in_range'] == int(Z['dist'].size), (s1, Z['dist'].size)
    assert 0 < s1['injected_cells'] <= n_missed < sa['injected_cells'], \
        (s1, sa)
    assert s1['selected_injected'] == s1['injected_cells']
    assert s1['restored_queries'] == s1['selected_injected']
    assert h1['restored_bad'] == 0 and h1['injected_selected'] == \
        s1['selected_injected']
    assert s2['selected_injected'] == s2['injected_cells'] == \
        s1['injected_cells']
    assert s2['restored_queries'] == 0                  # boost: no restore
    assert h2['restored_bad'] == s2['selected_injected']  # scores kept high
    json.dump({'n_missed': n_missed}, open(osp.join(out, 'n.json'), 'w'))
    print('       N_missed_D1=%d  E1/E2 missed_gt=%d  injected E1=%d all=%d'
          % (n_missed, s1['missed_gt'], s1['injected_cells'],
             sa['injected_cells']))


@case('exec2 E3: jitter run is deterministic for a fixed seed')
def test_jitter_determinism(tmp):
    a, _ = oracle(tmp, QD, 'j1', JIT)
    b, _ = oracle(tmp, QD, 'j2', JIT)
    assert a['metrics'] == b['metrics'] and a['stats'] == b['stats']
    assert a['args']['jitter_cells'] == 1 and a['args']['seed'] == 0


@case('exec2 M3: B0 tables equal direct computation; S9 pass/fail')
def test_b0(tmp):
    out = osp.join(tmp, 'recall_same_frames')
    npz = osp.join(out, 'recall_per_gt.npz')
    Z = np.load(npz)
    thr = list(Z['thr'])
    P = int(Z['P'])
    rc_, fc = Z['rank_ca'], Z['fin_ca']
    near, far = Z['dist'] < 30, Z['dist'] >= 30
    j = {t: thr.index(t) for t in (0.5, 1.0, 2.0)}
    g1 = (rc_[:, j[0.5]] >= P) & (rc_[:, j[1.0]] < P)
    want_fix = float(np.mean(fc[g1, j[0.5]])) if g1.any() else float('nan')
    rn = '%.5f' % np.mean(rc_[near, j[2.0]] < P)
    rf = '%.5f' % np.mean(rc_[far, j[2.0]] < P)
    od = osp.join(tmp, 'b0_ok')
    r, log = sh([PY, osp.join(QD, 'diag_b0_decoder.py'), '--npz', npz,
                 '--out-dir', od, '--s9-near', rn, '--s9-far', rf])
    assert r == 0, log
    res = json.load(open(osp.join(od, 'b0_decoder.json')))
    assert res['S9']['result'] == 'PASS'
    n_missed = json.load(open(osp.join(out, 'n.json')))['n_missed']
    assert res['N_missed_D1']['all'] == n_missed
    got = res['B0_2']['all']['G1']['fix_0.5']
    assert (got != got and want_fix != want_fix) or abs(got - want_fix) < 1e-12
    b01 = res['B0_1']['all']['thr']['2']
    assert abs(b01['init'] - np.mean(rc_[:, j[2.0]] < P)) < 1e-12
    assert abs(b01['final'] - np.mean(fc[:, j[2.0]])) < 1e-12
    r, log = sh([PY, osp.join(QD, 'diag_b0_decoder.py'), '--npz', npz,
                 '--out-dir', osp.join(tmp, 'b0_fail')])
    assert r == 1 and 'S9' in log and 'FAIL' in log, log


def synth_s9_npz(path):
    """npz whose R_near(200) / R_far(200) @2 m equal the #1 values."""
    n = 100000
    dist = np.r_[np.full(n, 10.0), np.full(n, 40.0)]
    hit = np.r_[np.arange(n) < 97319, np.arange(n) < 94944]
    rank = np.where(hit[:, None], 0, 1000).repeat(4, 1).astype(np.int64)
    fin = np.repeat(hit[:, None], 4, 1)
    np.savez_compressed(path, frame_counts=np.full(10, 20), thr=np.array(
        [0.5, 1.0, 2.0, 4.0]), ks=np.array([200, 300, 500, 1000]), P=200,
        class_names=np.array(['car'] * 10), class_range=True,
        dist=dist, label=np.zeros(2 * n, dtype=np.int64), rank_agn=rank,
        rank_ca=rank, fin_agn=fin, fin_ca=fin,
        frame=np.zeros(2 * n, dtype=np.int64))


def make_ref(tmp, cfg_opts):
    ref = json.load(open(REF1))
    ref['runtime']['config_md5'] = md5(CFG)
    ref['runtime']['checkpoint_md5'] = md5(CKPT)
    ref['runtime']['cfg_opts_env'] = cfg_opts
    path = osp.join(tmp, 'ref1', 'oracle_base.json')
    os.makedirs(osp.dirname(path), exist_ok=True)
    json.dump(ref, open(path, 'w'))
    return path, ref


@case("exec2 S0': real #1 md5s -> only M1-M5/LOG files changed; negatives")
def test_s0p(tmp):
    ex = osp.join(QD, 'diag_exec2_check.py')
    ref_path, ref = make_ref(tmp, 'X=1')
    od = osp.join(tmp, 's0p')
    rc, log = sh([PY, ex, 's0p', '--config', CFG, '--ckpt', CKPT,
                  '--ref-json', ref_path, '--out-dir', od],
                 env={'CFG_OPTS': 'X=1'})
    assert rc == 0, log
    res = json.load(open(osp.join(od, 's0p_check.json')))
    cats = {c['file']: c['category'] for c in res['changes']}
    assert 'OTHER' not in cats.values(), cats
    assert set(cats) == {'diag_oracle_query.py', 'diag_b0_decoder.py',
                         'run_diag.sh', 'exec02_env.sh',
                         'diag_exec2_check.py', 'oracle_breakdown.py',
                         'diag_summarize.py'}, sorted(cats)
    for f, c in sorted(cats.items()):
        print('       %-24s %s' % (f, c))
    rc, log = sh([PY, ex, 's0p', '--config', CFG, '--ckpt', CKPT,
                  '--ref-json', ref_path, '--out-dir', od],
                 env={'CFG_OPTS': 'X=2'})
    assert rc == 1 and 'CFG_OPTS differs' in log, log
    bad = copy.deepcopy(ref)
    bad['runtime']['diag_files_md5']['diag_common.py'] = '0' * 32
    bp = osp.join(tmp, 'ref1', 'bad.json')
    json.dump(bad, open(bp, 'w'))
    rc, log = sh([PY, ex, 's0p', '--config', CFG, '--ckpt', CKPT,
                  '--ref-json', bp, '--out-dir', od], env={'CFG_OPTS': 'X=1'})
    assert rc == 1 and 'diag_common.py CHANGED is not covered' in log, log


def fake_oracle_json(path, tag, base_metrics, d_map, stats, extra_args):
    m = dict(base_metrics)
    pre = 'NuScenes metric/pred_instances_3d_NuScenes/'
    for k in list(m):
        if k.startswith(pre) and ('_AP_dist_' in k or k.endswith('/mAP')):
            m[k] = float(m[k]) + d_map
    args = {'tag': tag, 'oracle': 'loc', 'subset': 'missed', 'score': 'orig',
            'range': [0.0, 1000.0], 'jitter_cells': 0, 'seed': 0,
            'miss_thr': 2.0}
    args.update(extra_args)
    os.makedirs(osp.dirname(path), exist_ok=True)
    json.dump({'args': args, 'stats': stats, 'metrics': m,
               'runtime': {'elapsed_s': 1.0}}, open(path, 'w'))


@case('exec2 M5: summary deltas, retention, S5\'-S8, breakdown, no Q verdict')
def test_summary(tmp):
    root = osp.join(tmp, 'exec2_root')
    base = json.load(open(REF1))['metrics']
    st = {'injected_cells': 100.0, 'selected_injected': 100.0,
          'restored_queries': 100.0, 'capped_cells': 0.0,
          'init_mismatch_rate': 0.0, 'missed_gt': 3950.0,
          'gt_in_range': 118538.0, 'frames': 6019.0}
    fake_oracle_json(osp.join(root, 'oracle_e0_base', 'oracle_e0_base.json'),
                     'e0_base', base, 0.0, dict(st, injected_cells=0.0,
                                                selected_injected=0.0,
                                                restored_queries=0.0),
                     {'oracle': 'none', 'subset': 'all'})
    fake_oracle_json(osp.join(root, 'oracle_e1_missed_orig',
                              'oracle_e1_missed_orig.json'),
                     'e1_missed_orig', base, 0.004, st, {})
    fake_oracle_json(osp.join(root, 'oracle_e2_missed_boost',
                              'oracle_e2_missed_boost.json'),
                     'e2_missed_boost', base, 0.009,
                     dict(st, restored_queries=0.0), {'score': 'boost'})
    fake_oracle_json(osp.join(root, 'oracle_e3_jitter',
                              'oracle_e3_jitter.json'),
                     'e3_jitter', base, 0.0045,
                     dict(st, injected_cells=117000.0,
                          selected_injected=117000.0,
                          restored_queries=117000.0),
                     {'subset': 'all', 'jitter_cells': 1})
    json.dump({'result': 'PASS', 'diff': {'mAP': 0.0003, 'NDS': 0.0001}},
              open(osp.join(root, 'oracle_e0_base', 's5p_gate.json'), 'w'))
    json.dump({'result': 'PASS'}, open(osp.join(root, 's0p_check.json'),
                                       'w'))
    json.dump({'S9': {'result': 'PASS'}, 'N_missed_D1': {'all': 3960}},
              open(osp.join(root, 'b0_decoder.json'), 'w'))
    rc, log = sh([PY, osp.join(QD, 'diag_exec2_check.py'), 'summary',
                  '--root', root])
    assert rc == 0, log
    s = json.load(open(osp.join(root, 'exec2_summary.json')))
    assert abs(s['coverage_orig_pp'] - 0.4) < 1e-9, s['coverage_orig_pp']
    assert abs(s['coverage_boost_pp'] - 0.9) < 1e-9
    assert abs(s['jitter_retention'] - 0.45 / 1.875) < 1e-9
    res = {(c['id'], c['desc']): c['result'] for c in s['self_checks']}
    assert all(v == 'PASS' for v in res.values()), res
    assert s['self_checks_all_pass'] is True
    txt = open(osp.join(root, 'exec2_summary.txt')).read()
    assert 'Class-mean AP per match threshold' in txt
    for word in ('Q1', 'Q3', 'Dome-DETR', 'continue', 'stop'):
        assert word not in txt, word
    json.dump({'S9': {'result': 'PASS'}, 'N_missed_D1': {'all': 4100}},
              open(osp.join(root, 'b0_decoder.json'), 'w'))
    sh([PY, osp.join(QD, 'diag_exec2_check.py'), 'summary', '--root', root])
    s = json.load(open(osp.join(root, 'exec2_summary.json')))
    s8 = [c for c in s['self_checks'] if c['id'] == 'S8' and
          'N_missed_D1' in c['desc']]
    assert all(c['result'] == 'FAIL' for c in s8), s8    # 3.7% off


@case('exec2 M4: run_diag b0 / exec2 gating, order, refs, single GPU')
def test_run_diag_exec2(tmp):
    b = T.stub_env(tmp)
    stub = open(osp.join(b, 'python')).read().replace(
        '*diag_gate_s5.py*|*diag_merge_bins.py*)',
        '*diag_gate_s5.py*|*diag_merge_bins.py*|*diag_exec2_check.py*|'
        '*diag_b0_decoder.py*)')
    open(osp.join(b, 'python'), 'w').write(stub)
    out = osp.join(tmp, 'rd2_out')
    log = osp.join(tmp, 'stub2.log')
    ref_path, _ = make_ref(tmp, 'X=1')
    os.makedirs(osp.join(out, 'oracle_base'))
    shutil.copy(ref_path, osp.join(out, 'oracle_base', 'oracle_base.json'))
    os.makedirs(osp.join(out, 'recall'))
    synth_s9_npz(osp.join(out, 'recall', 'recall_per_gt.npz'))
    base = {'PATH': b + os.pathsep + os.environ['PATH'], 'STUB_LOG': log,
            'REAL_PY': PY, 'OUT': out, 'CONFIG': CFG, 'CKPT': CKPT,
            'CFG_OPTS': 'X=1', 'STUB_MAP': '0.665463',
            'STUB_NDS': '0.706154', 'CUDA_VISIBLE_DEVICES': '2'}
    rdsh = osp.join(QD, 'run_diag.sh')

    def run(step, **env):
        e = dict(base)
        e.update(env)
        if osp.exists(log):
            os.remove(log)
        rc, txt = sh(['bash', rdsh, step], env=e)
        calls = [c.split(' ', 1)[1] for c in open(log).read().splitlines()] \
            if osp.exists(log) else []
        return rc, txt, calls

    def tags(calls):
        return [c.split('--tag ')[1].split()[0] for c in calls
                if 'diag_oracle_query.py' in c]

    rc, txt, _ = run('exec2', BASE_MAP='0.665463', BASE_NDS='0.706154')
    assert rc == 2 and "run 'b0' first" in txt, txt
    rc, txt, calls = run('b0')
    assert rc == 0, txt
    assert "S0'  PASS" in txt and 'S9' in txt and 'PASS' in txt, txt
    assert osp.isfile(osp.join(out, 'exec2', 'b0_decoder.json'))
    rc, txt, calls = run('exec2', BASE_MAP='0.6648', BASE_NDS='0.7060')
    assert rc == 2 and 'exec02_env.sh' in txt and not tags(calls), txt
    rc, txt, calls = run('exec2', BASE_MAP='0.665463', BASE_NDS='0.706154',
                         CUDA_VISIBLE_DEVICES='1,2')
    assert rc == 2 and 'exactly 1' in txt and not tags(calls), txt
    rc, txt, calls = run('exec2', BASE_MAP='0.665463', BASE_NDS='0.706154',
                         STUB_MAP='0.6700')
    assert rc == 4 and tags(calls) == ['e0_base'] and 'NOT run' in txt, txt
    rc, txt, calls = run('exec2', BASE_MAP='0.665463', BASE_NDS='0.706154',
                         GPUS='2')
    assert rc == 0, txt
    assert tags(calls) == ['e0_base', 'e1_missed_orig', 'e2_missed_boost',
                           'e3_jitter'], tags(calls)
    oc = [c for c in calls if 'diag_oracle_query.py' in c]
    assert '--subset missed --range 0 1000 --score orig' in oc[1]
    assert '--subset missed --range 0 1000 --score boost' in oc[2]
    assert '--subset all --range 0 1000 --score orig --jitter-cells 1 ' \
        '--seed 0' in oc[3]
    assert all('--work-dir %s/exec2/oracle_' % out in c for c in oc)
    assert all('--launcher' not in c and 'torchrun' not in c for c in oc)
    assert osp.isfile(osp.join(out, 'exec2', 'exec2_summary.json'))
    assert osp.isfile(osp.join(out, 'exec2', 'oracle_e0_base',
                               's5p_gate.json'))
    logs = [f for f in os.listdir(osp.join(out, 'logs'))
            if f.startswith('exec2_')]
    assert logs, os.listdir(osp.join(out, 'logs'))


@case('exec02_env.sh: exec #1 inputs with exec #2 BASE references')
def test_env2(tmp):
    ep, _ = T.make_fake_ep(osp.join(tmp, 'e2'))
    ws = T.make_workspace(osp.join(tmp, 'e2'), 'ws')
    for f in ('exec01_env.sh', 'exec02_env.sh'):
        shutil.copy(osp.join(QD, f), osp.join(ws, 'tools', 'query_diag', f))
    rc, out = sh(['bash', '-c', 'source tools/query_diag/exec02_env.sh && '
                  'echo "CK=$CKPT B=$BASE_MAP/$BASE_NDS"'],
                 env={'EP_REPO': ep}, cwd=ws)
    assert rc == 0 and 'B=0.665463/0.706154' in out, out
    assert 'CK=%s/work_dirs/exec01_inputs/epoch_5.pth' % ws in out, out
    rc, out = sh(['bash', '-c', 'source tools/query_diag/exec02_env.sh; '
                  'echo "rc=$?"'], env={'EP_REPO': ep},
                 cwd=osp.join(ep, 'work_dirs'))
    assert 'rc=1' in out, out


# ---------------------------------------------------------------------------
# E2-1 (EXEC02_DECISIONS.md): the #1 logging changes must not change any
# result. Proof: rerun with exec doc #1 data / steps, #1 code vs current code.
# ---------------------------------------------------------------------------

OLD_SUMMARIZE = '484b6bb'        # diag_summarize.py md5 a86cf0a3 (#1 runs)
OLD_RUN_DIAG = 'de1bbcd'         # run_diag.sh md5 e19bf9b4 (#1 runs)
RES1 = osp.join(REPO, 'projects/DomeDETR/results/exec01')
EVID = []


def git_show(commit, rel):
    return subprocess.check_output(['git', 'show', '%s:%s' % (commit, rel)],
                                   cwd=REPO)


TIME_FIELD = re.compile(rb'"time": "\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"')


def tree(root, skip=()):
    """{relpath: md5} of every file under root (symlinks by target). The
    wall-clock "time" field that diag_gate_s5.py / diag_merge_bins.py write
    into their json runtime block is masked; nothing else is."""
    out = {}
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d != '__pycache__']
        for n in fn:
            p = osp.join(dp, n)
            rel = osp.relpath(p, root)
            if skip and rel.startswith(skip):
                continue
            if osp.islink(p):
                out[rel] = 'L:' + os.readlink(p)
            else:
                data = TIME_FIELD.sub(b'"time": "<t>"', open(p, 'rb').read())
                out[rel] = hashlib.md5(data).hexdigest()
    return out


def exec1_out(out):
    """exec doc #1 step 2/3 returned files in the server layout."""
    for tag in ('base', 'all', 'near', 'far', 'all_cls'):
        d = osp.join(out, 'oracle_' + tag)
        os.makedirs(d)
        shutil.copy(osp.join(RES1, 'step3_oracle', 'oracle_%s.json' % tag), d)
    shutil.copy(osp.join(RES1, 'step3_oracle', 's5_gate.json'),
                osp.join(out, 'oracle_base'))
    os.makedirs(osp.join(out, 'recall'))
    shutil.copy(osp.join(RES1, 'step2_recall_summary.json'),
                osp.join(out, 'recall', 'recall_summary.json'))


@case('E2-1 LOG: diag_summarize.py on #1 data == archive; only +summary.txt')
def test_summarize_equivalence(tmp):
    ref = json.load(open(REF1))['runtime']['diag_files_md5']
    old = osp.join(tmp, 'diag_summarize_exec1.py')
    open(old, 'wb').write(git_show(OLD_SUMMARIZE,
                                   'tools/query_diag/diag_summarize.py'))
    assert md5(old) == ref['diag_summarize.py']
    arch_path = osp.join(RES1, 'step3_summary_reconstructed.txt')
    arch = open(arch_path, 'rb').read()
    new = osp.join(QD, 'diag_summarize.py')
    res = {}
    for ver, script in (('old', old), ('new', new)):
        ws = osp.join(tmp, 'sum_ws_' + ver)
        exec1_out(osp.join(ws, 'work_dirs', 'diag'))
        before = tree(ws)
        p = subprocess.run([PY, script, 'work_dirs/diag'], cwd=ws,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert p.returncode == 0 and not p.stderr, p.stderr
        res[ver] = (p.stdout, before, tree(ws), ws)
    so, bo, ao, _ = res['old']
    sn, bn, an, wsn = res['new']
    save = b'[summary] saved work_dirs/diag/summary.txt\n'
    assert so == arch, 'exec1 code does not reproduce the archive'
    assert sn == arch + save, sn[-300:]
    assert ao == bo                                   # #1 code writes nothing
    assert {k: v for k, v in an.items() if k in bn} == bn  # inputs untouched
    assert sorted(set(an) - set(bn)) == ['work_dirs/diag/summary.txt']
    txt = open(osp.join(wsn, 'work_dirs/diag/summary.txt'), 'rb').read()
    assert txt == arch
    EVID.extend([
        'diag_summarize.py  #1 (%s, md5 %s) vs current (md5 %s)'
        % (OLD_SUMMARIZE, md5(old)[:8], md5(new)[:8]),
        '  input: results/exec01 step3_oracle/*.json, s5_gate.json, '
        'step2_recall_summary.json',
        '  #1 code stdout      == step3_summary_reconstructed.txt '
        '(%d bytes, md5 %s)' % (len(arch), md5(arch_path)[:8]),
        '  current stdout      == archive + 1 line: %r' % save.decode(),
        '  current summary.txt == archive (byte-identical)',
        '  files changed by current code: +work_dirs/diag/summary.txt only; '
        'inputs unchanged; #1 code: no files written'])


TORCHRUN = '#!/usr/bin/env bash\nexec python "$@"\n'


def run_clean(cmd, env, cwd):
    """Run with ONLY the given environment (nothing leaks from the caller)."""
    p = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True)
    return p.returncode, p.stdout


def short(rel):
    rel = rel.replace('work_dirs/diag/', '')
    return re.sub(r'_\d{8}_\d{6}\.log$', '_<time>.log', rel)


@case('E2-1 LOG: run_diag.sh #1 vs current on every #1 step: same calls/rc')
def test_run_diag_log_equivalence(tmp):
    ref = json.load(open(REF1))['runtime']['diag_files_md5']
    old_root = osp.join(tmp, 'qd_exec1')
    os.makedirs(old_root)
    tar = subprocess.check_output(['git', 'archive', OLD_RUN_DIAG,
                                   'tools/query_diag'], cwd=REPO)
    subprocess.run(['tar', '-x', '-C', old_root], input=tar, check=True)
    old_qd = osp.join(old_root, 'tools', 'query_diag')
    got = {f: md5(osp.join(old_qd, f)) for f in os.listdir(old_qd)
           if f.endswith(('.py', '.sh', '.json'))}
    assert got == ref, 'de1bbcd tools/query_diag != #1 runtime md5 set'
    b = osp.join(tmp, 'eqbin')
    os.makedirs(b)
    stub = T.PYSTUB.replace('*diag_gate_s5.py*|*diag_merge_bins.py*)',
                            '*diag_gate_s5.py*|*diag_merge_bins.py*|'
                            '*diag_summarize.py*)')
    for name, body in (('nvidia-smi', T.NVSMI), ('python', stub),
                       ('torchrun', TORCHRUN)):
        open(osp.join(b, name), 'w').write(body)
        os.chmod(osp.join(b, name), 0o755)
    tpl = T.make_workspace(tmp, 'eq_tpl')
    shutil.rmtree(osp.join(tpl, 'tools', 'query_diag'))
    t_exec1 = osp.join(tmp, 'out_exec1')
    exec1_out(t_exec1)
    t_rec = osp.join(tmp, 'out_recall')
    rc, log = sh([PY, MOCK, 'recall', QD, osp.join(t_rec, 'recall'), CFG,
                  CKPT])
    assert rc == 0, log[-2000:]
    slog = osp.join(tmp, 'eq_stub.log')
    base = {'PATH': b + os.pathsep + os.environ['PATH'], 'STUB_LOG': slog,
            'REAL_PY': PY, 'STUB_MAP': '0.6651', 'STUB_NDS': '0.6962'}
    run1 = dict(CONFIG=CFG, CKPT=CKPT, CUDA_VISIBLE_DEVICES='2')
    gate = dict(run1, BASE_MAP='0.6651', BASE_NDS='0.6962')
    ws = osp.join(tmp, 'eq_ws')              # same path for both versions
    scenarios = [
        ('build-ops', {}, None, 'in-place op build'),
        ('s0', dict(CONFIG=CFG, CKPT=CKPT, BASE_MAP='0.6648',
                    BASE_NDS='0.7060', CFG_OPTS='a.b=1',
                    TRAIN_WORK_DIR='work_dirs/exec01_inputs/train_run'),
         None, 'S0 checker call'),
        ('quick', dict(run1, CFG_OPTS='a.b=1 c=x'), None, '300-frame smoke'),
        ('quick', {}, None, 'refused: CONFIG unset'),
        ('recall', run1, None, 'D1 full run'),
        ('recall', dict(run1, STUB_APPS_2='4242, python, 9000 MiB'), None,
         'refused: C5, GPU busy'),
        ('oracle', gate, None, '5 groups, 1 GPU, S5 PASS'),
        ('oracle', dict(gate, CUDA_VISIBLE_DEVICES='1,2', GPUS='2'), None,
         '5 groups, 2 GPUs (torchrun), S5 PASS'),
        ('oracle', dict(gate, STUB_MAP='0.6700'), None,
         'S5 FAIL: stops after base'),
        ('s5-adjudicate', dict(gate, STUB_MAP='0.6653'), t_exec1,
         'tools/test.py vs #1 oracle_base.json'),
        ('merge', {}, t_rec, 'D1 merge (mock npz)'),
        ('summary', {}, t_exec1, '#1 step 3 / step 2 data'),
        ('merge', dict(PROTECTED_DIRS=osp.join(ws, 'work_dirs')), t_rec,
         'refused: OUT protected'),
    ]
    EVID.append('run_diag.sh  #1 (%s, md5 %s; the whole #1 tools/query_diag, '
                'md5 set == #1 runtime) vs current (md5 %s)'
                % (OLD_RUN_DIAG, md5(osp.join(old_qd, 'run_diag.sh'))[:8],
                   md5(osp.join(QD, 'run_diag.sh'))[:8]))
    EVID.append('  same workspace path, OUT and stubbed nvidia-smi / python / '
                'torchrun for both; compared: exit code, every external call '
                '(argv), terminal text, every file (md5; only the wall-clock '
                '"time" field of the gate / merge json is masked)')
    for step, env, out_tpl, desc in scenarios:
        r = {}
        for ver, qd in (('old', old_qd), ('new', QD)):
            shutil.rmtree(ws, ignore_errors=True)
            shutil.copytree(tpl, ws, symlinks=True)
            shutil.copytree(qd, osp.join(ws, 'tools', 'query_diag'),
                            ignore=shutil.ignore_patterns('__pycache__'))
            if out_tpl:
                shutil.copytree(out_tpl, osp.join(ws, 'work_dirs', 'diag'))
            if osp.exists(slog):
                os.remove(slog)
            rc, txt = run_clean(['bash', 'tools/query_diag/run_diag.sh',
                                 step], dict(base, **env), ws)
            calls = open(slog).read() if osp.exists(slog) else ''
            files = tree(ws, skip=osp.join('tools', 'query_diag'))
            logs = sorted(k for k in files
                          if k.startswith('work_dirs/diag/logs/'))
            logtxt = [open(osp.join(ws, k)).read() for k in logs]
            r[ver] = (rc, txt, calls, files, logs, logtxt)
        rc0, t0, c0, f0, l0, _ = r['old']
        rc1, t1, c1, f1, l1, lt1 = r['new']
        tag = '%s (%s)' % (step, desc)
        assert rc0 == rc1, (tag, rc0, rc1, t0, t1)
        assert c0 == c1, (tag, c0, c1)
        assert not l0, (tag, l0)
        if l1:
            assert len(l1) == 1, l1
            assert l1[0].startswith('work_dirs/diag/logs/%s_' % step), l1
            first, rest = t1.split('\n', 1)
            assert first == '[run_diag] step=%s log=%s' % (step, l1[0]), first
            how = 'old == new minus 1 line, log == terminal'
            if step == 'summary':          # diag_summarize.py change, E2-1
                t0 += '[summary] saved work_dirs/diag/summary.txt\n'
                how = ('old == new minus 1 line + "[summary] saved" line, '
                       'log == terminal')
            assert rest == t0, (tag, t0, rest)
            assert lt1[0] == t1, (tag, lt1[0], t1)    # log == terminal
            extra = {l1[0]}
            if step == 'summary':
                extra.add('work_dirs/diag/summary.txt')
            assert set(f1) - set(f0) == extra, (tag, set(f1) ^ set(f0))
            assert all(f1[k] == f0[k] for k in f0), \
                (tag, [k for k in f0 if f1[k] != f0[k]])
        else:
            assert t0 == t1 and f0 == f1, (tag, t0, t1)
            extra, how = set(), 'old == new (stops before logging)'
        n = len(c0.splitlines())
        add = ', '.join(sorted(short(k) for k in extra)) or '-'
        print('       %-14s rc=%d calls=%d  %s  +%s'
              % (step, rc0, n, how, add))
        EVID.append('  %-52s rc=%d calls=%d identical | terminal: %s | '
                    'files: +%s' % (tag, rc0, n, how, add))


@case('E2-3: missed threshold follows head.num_proposals (mock P=40)')
def test_p_from_head(tmp):
    env = {'QD_MOCK_FRAMES': '12', 'QD_MOCK_SEED': '1', 'QD_MOCK_P': '40'}
    out = osp.join(tmp, 'recall_p40')
    rc, log = sh([PY, MOCK, 'recall', QD, out, CFG, CKPT], env=env)
    assert rc == 0, log[-2000:]
    Z = np.load(osp.join(out, 'recall_per_gt.npz'))
    ti = list(Z['thr']).index(2.0)
    assert int(Z['P']) == 40
    n40 = int((Z['rank_ca'][:, ti] >= 40).sum())
    n200 = int((Z['rank_ca'][:, ti] >= 200).sum())
    assert n40 > n200, (n40, n200)
    e, _ = oracle(tmp, QD, 'p40', LOC + ['--subset', 'missed', '--score',
                                         'orig'], env)
    assert e['num_proposals'] == 40, e['num_proposals']
    assert e['stats']['missed_gt'] == n40, (e['stats'], n40, n200)
    print('       P=40: missed_gt=%d == D1 rank_ca>=40 (%d); rank_ca>=200 '
          'would give %d' % (e['stats']['missed_gt'], n40, n200))


CFG = CKPT = None


def main():
    global CFG, CKPT
    tmp = tempfile.mkdtemp(prefix='qd_exec2_')
    CFG = osp.join(tmp, 'fake_config.py')
    CKPT = osp.join(tmp, 'fake_ckpt.pth')
    open(CFG, 'w').write('# fake\n')
    open(CKPT, 'wb').write(os.urandom(1024))
    test_subset_all_identical(tmp)
    test_missed(tmp)
    test_jitter_determinism(tmp)
    test_b0(tmp)
    test_s0p(tmp)
    test_summary(tmp)
    test_run_diag_exec2(tmp)
    test_env2(tmp)
    test_summarize_equivalence(tmp)
    test_run_diag_log_equivalence(tmp)
    test_p_from_head(tmp)
    if '--evidence' in sys.argv:
        path = sys.argv[sys.argv.index('--evidence') + 1]
        with open(path, 'w') as f:
            f.write('\n'.join(EVID) + '\n')
        print('evidence written to', path)
    n_fail = sum(1 for r in T.RESULTS if r[1] != 'PASS')
    print('\n%d tests, %d failed' % (len(T.RESULTS), n_fail))
    shutil.rmtree(tmp, ignore_errors=True)
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
