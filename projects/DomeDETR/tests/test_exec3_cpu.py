# -*- coding: utf-8 -*-
"""Dev-side CPU tests for exec doc #3 step 0 tools (projects/QueryLoc);
not deployed. Run from the repo root (needs torch, numpy, mmengine):
  python projects/DomeDETR/tests/test_exec3_cpu.py
"""
import hashlib
import importlib.util
import itertools
import json
import math
import os
import os.path as osp
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import test_query_diag_cpu as T  # noqa: E402

QD, MOCK, PY, REPO = T.QD, T.MOCK, T.PY, T.REPO
QL = osp.join(REPO, 'projects', 'QueryLoc')
QLT = osp.join(QL, 'tools')
CHECK = osp.join(QLT, 'queryloc_check.py')
E0 = osp.join(REPO, 'projects/DomeDETR/results/exec02/step2_oracle/'
              'oracle_e0_base.json')
case, sh = T.case, T.sh
CFG = CKPT = None


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SC = load('ql_subcell', osp.join(QL, 'queryloc', 'subcell.py'))


def gauss_map(H, W, C, ch, r0, c0, prec, p0=0.8, floor=1e-4):
    """heat[ch] = p0 exp(-0.5 v^T prec v), v = cell centre - (r0, c0)."""
    heat = torch.full((1, C, H, W), floor, dtype=torch.float64)
    A = torch.tensor(prec, dtype=torch.float64)
    for r in range(H):
        for c in range(W):
            v = torch.tensor([r + 0.5 - r0, c + 0.5 - c0],
                             dtype=torch.float64)
            heat[0, ch, r, c] = p0 * math.exp(-0.5 * float(v @ A @ v))
    return heat


@case('3 subcell: DARK exact on log-quadratic peaks (incl. cross term)')
def test_dark_exact(tmp):
    H = W = 11
    for dr, dc, prec in ((0.31, -0.22, [[1.0, 0.3], [0.3, 0.6]]),
                         (-0.49, 0.45, [[0.5, 0.0], [0.0, 0.5]]),
                         (0.0, 0.0, [[2.0, -0.5], [-0.5, 1.0]]),
                         (0.12, 0.4, [[0.3, 0.1], [0.1, 3.0]])):
        r, c = 5, 6
        heat = gauss_map(H, W, 3, 2, r + 0.5 + dr, c + 0.5 + dc, prec)
        idx = torch.tensor([[r * W + c]])
        nb, valid, border = SC.neighborhood3x3(heat, idx,
                                               torch.tensor([[2]]))
        off, st, cl = SC.dark_offsets(nb, border)
        assert int(st) == SC.OK and not bool(cl), (st, cl)
        assert abs(float(off[0, 0, 0]) - dr) < 1e-6 and \
            abs(float(off[0, 0, 1]) - dc) < 1e-6, (off, dr, dc)
        assert torch.equal(nb[0, 0], heat[0, 2, r - 1:r + 2, c - 1:c + 2])
    # true maximum outside +-0.5 (strong anisotropy): clipped
    heat = gauss_map(H, W, 1, 0, 5.5 + 0.9, 5.5, [[0.05, 0.0], [0.0, 1.0]])
    nb, valid, border = SC.neighborhood3x3(heat, torch.tensor([[5 * W + 5]]),
                                           torch.tensor([[0]]))
    off, st, cl = SC.dark_offsets(nb, border)
    assert int(st) == SC.OK and bool(cl) and abs(float(off[0, 0, 0]) - 0.5) \
        < 1e-9, (off, st, cl)


@case('3 subcell: fallbacks (flat, saddle, border); soft-argmax')
def test_dark_fallback(tmp):
    H = W = 7
    flat = torch.full((1, 1, H, W), 0.3)
    idx, cls = torch.tensor([[3 * W + 3]]), torch.tensor([[0]])
    nb, valid, border = SC.neighborhood3x3(flat, idx, cls)
    off, st, _ = SC.dark_offsets(nb, border)
    assert int(st) == SC.NOT_NEG_DEF and float(off.abs().sum()) == 0.0
    saddle = torch.full((1, 1, H, W), 0.1)
    saddle[0, 0, 3, 2] = saddle[0, 0, 3, 4] = 0.5      # rises along col
    saddle[0, 0, 3, 3] = 0.3
    nb, valid, border = SC.neighborhood3x3(saddle, idx, cls)
    off, st, _ = SC.dark_offsets(nb, border)
    assert int(st) == SC.NOT_NEG_DEF, st
    heat = gauss_map(H, W, 1, 0, 0.6, 3.4, [[1.0, 0.0], [0.0, 1.0]])
    nb, valid, border = SC.neighborhood3x3(heat, torch.tensor([[0 * W + 3]]),
                                           cls)
    off, st, _ = SC.dark_offsets(nb, border)
    assert bool(border) and int(st) == SC.BORDER and \
        float(off.abs().sum()) == 0.0
    assert not bool(valid[0, 0, 0].any()) and bool(valid[0, 0, 1:].all())
    s = SC.softargmax_offsets(nb, valid)
    assert float(s[0, 0, 0]) >= 0.0          # off-map row excluded
    sym = torch.full((1, 1, H, W), 0.01)
    sym[0, 0, 2:5, 2:5] = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1.]])
    nb, valid, _ = SC.neighborhood3x3(sym, idx, cls)
    assert float(SC.softargmax_offsets(nb, valid).abs().max()) < 1e-7
    sym[0, 0, 3, 4] = 3.0
    nb, valid, _ = SC.neighborhood3x3(sym, idx, cls)
    s = SC.softargmax_offsets(nb, valid)
    assert float(s[0, 0, 1]) > 0 and abs(float(s[0, 0, 0])) < 1e-7, s
    assert torch.equal(SC.rowcol_to_xy(torch.tensor([1., 2.]), True),
                       torch.tensor([2., 1.]))


def run_0a(tmp, name, env, extra=()):
    out = osp.join(tmp, name)
    rc, log = sh([PY, MOCK, 'script', osp.join(QLT, 'subcell_0a.py'), CFG,
                  CKPT, '--out-dir', out] + list(extra), env=env)
    res = json.load(open(osp.join(out, 'subcell_0a.json'))) \
        if osp.isfile(osp.join(out, 'subcell_0a.json')) else None
    return rc, log, res, out


@case('3 step 0a on the Gaussian mock: conventions, S-0a1 == D1 recall')
def test_0a_mock(tmp):
    env = {'QD_MOCK_GAUSS': '1', 'QD_MOCK_FRAMES': '8', 'QD_MOCK_SEED': '3'}
    rec = osp.join(tmp, 'rec_gauss')
    rc, log = sh([PY, MOCK, 'recall', QD, rec, CFG, CKPT], env=env)
    assert rc == 0, log[-2000:]
    Z = np.load(osp.join(rec, 'recall_per_gt.npz'))
    j = list(Z['thr']).index(0.5)
    d1 = float(np.mean(Z['rank_ca'][:, j] < int(Z['P'])))
    rc, log, res, out = run_0a(tmp, 'sub0a', env,
                               ['--b0-init05', repr(d1), '--b0-tol', '1e-12'])
    assert rc == 0 and res is not None, log[-3000:]
    chk = {c['id']: c for c in res['self_checks']}
    assert chk['S-0a1']['result'] == 'PASS', chk
    assert chk['S-0a2']['value'] == 1.0, chk
    S = res['summary']
    A = np.load(osp.join(out, 'subcell_0a_per_gt.npz'))
    att = (A['group'] == 'G0') | (A['group'] == 'G1')
    exact = att & (A['status'] == SC.OK)
    assert exact.sum() >= 0.9 * att.sum() > 50, (exact.sum(), att.sum())
    frac = float(np.mean(A['d_dark'][exact] < 1e-3))
    assert frac >= 0.95, frac          # DARK recovers the true centre
    moved = A['d_cell'][exact] > 1e-3          # distractor spikes may
    closer = A['d_dark'][exact][moved] < A['d_cell'][exact][moved]
    assert closer.mean() >= 0.98, closer.mean()  # disturb a few patches
    r = S['recall']['all']
    assert abs(r['cell'] - d1) < 1e-12
    assert r['dark'] >= r['cell'] and r['soft'] >= r['cell'] - 1e-12, r
    assert S['trigger_0b']['triggered'] == (r['d_dark_minus_cell_pp'] >= 0.5)
    txt = open(osp.join(out, 'subcell_0a.txt')).read()
    assert 'Step 0b trigger' in txt and 'S-0a1' in txt
    print('       mock: N_att=%d exact=%d d_dark<1mm %.3f | recall@0.5 cell '
          '%.4f dark %.4f soft %.4f' % (att.sum(), exact.sum(), frac,
                                        r['cell'], r['dark'], r['soft']))
    # spikes only (original mock): no sub-cell info, but the run completes
    env2 = {'QD_MOCK_FRAMES': '4', 'QD_MOCK_SEED': '3'}
    rc, log, res, _ = run_0a(tmp, 'sub0a_spike', env2,
                             ['--b0-init05', '0.0'])
    assert rc == 1 and res is not None, log[-2000:]   # S-0a1 FAIL as asked
    chk = {c['id']: c['result'] for c in res['self_checks']}
    assert chk == {'S-0a1': 'FAIL', 'S-0a2': 'PASS'}, chk


def fake_ref(tmp, name, cfg_opts):
    r = json.load(open(E0))
    r['runtime']['config_md5'] = _md5(CFG)
    r['runtime']['checkpoint_md5'] = _md5(CKPT)
    r['runtime']['cfg_opts_env'] = cfg_opts
    p = osp.join(tmp, name + '.json')
    json.dump(r, open(p, 'w'))
    return p


def _md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def copy_root(tmp, name):
    root = osp.join(tmp, name)
    for rel in ('projects/BEVFusion', 'projects/QueryLoc',
                'tools/query_diag'):
        shutil.copytree(osp.join(REPO, rel), osp.join(root, rel),
                        ignore=shutil.ignore_patterns('__pycache__'))
    return root


@case("3 S0''': repo == #2 record / manifest / registry; negatives")
def test_s0ppp(tmp):
    ref = fake_ref(tmp, 'ref_e0', 'X=1')
    out = osp.join(tmp, 's0ppp_out')
    base = [PY, CHECK, 's0ppp', '--config', CFG, '--ckpt', CKPT,
            '--ref-json', ref, '--out-dir', out]
    rc, log = sh(base, env={'CFG_OPTS': 'X=1'})
    assert rc == 0 and "S0'''  PASS" in log, log
    assert '33 / 33 manifest files identical' in log, log
    rc, log = sh(base, env={'CFG_OPTS': 'X=2'})
    assert rc == 1 and 'CFG_OPTS differs' in log, log
    root = copy_root(tmp, 'neg_root')
    with open(osp.join(root, 'projects/QueryLoc/queryloc/subcell.py'),
              'a') as f:
        f.write('\n')
    open(osp.join(root, 'projects/BEVFusion/bevfusion/extra_mod.py'),
         'w').write('x = 1\n')
    with open(osp.join(root, 'projects/BEVFusion/DESIGN.md'), 'a') as f:
        f.write('\nnote\n')
    with open(osp.join(root, 'tools/query_diag/diag_common.py'), 'a') as f:
        f.write('\n')
    os.makedirs(osp.join(root, 'projects/BEVFusion/bevfusion/ops/x/build'))
    open(osp.join(root, 'projects/BEVFusion/bevfusion/ops/x/k.cpython-38-'
                  'x86_64-linux-gnu.so'), 'w').write('')
    shutil.copy(osp.join(REPO, 'tools/query_diag/s0_manifest_82c2b4e.json'),
                osp.join(root, 'tools/query_diag'))
    rc, log = sh(base + ['--root', root], env={'CFG_OPTS': 'X=1'})
    assert rc == 1, log
    for s in ('queryloc/subcell.py CHANGED', 'extra_mod.py EXTRA',
              'diag_common.py CHANGED', 'WARN projects/BEVFusion: '
              'projects/BEVFusion/DESIGN.md DIFF'):
        assert s in log, (s, log)
    assert '.so' not in log.split('FAIL')[-1], log


PYSTUB = r'''#!/usr/bin/env bash
echo "python $*" >> "$STUB_LOG"
case "$*" in
  *queryloc_check.py*) exec "$REAL_PY" "$@" ;;
esac
if [[ "$*" == *subcell_0a.py* ]]; then
  args=("$@"); out=""
  for ((i=0; i<${#args[@]}; i++)); do
    [ "${args[$i]}" = --out-dir ] && out=${args[$((i+1))]}
  done
  mkdir -p "$out"; echo '{}' > "$out/subcell_0a.json"
  exit "${STUB_0A_RC:-0}"
fi
exit "${STUB_RC:-0}"
'''
TORCHRUN = '#!/usr/bin/env bash\necho "torchrun $*" >> "$STUB_LOG"\nexit 0\n'


@case('3 run_queryloc.sh: probe / s0ppp / 0a / check-load / sm0 / sm1 / '
      '0b / rehearse: order, exact arguments, gating, C5, logs')
def test_runner(tmp):
    b = osp.join(tmp, 'ql_bin')
    os.makedirs(b)
    for name, body in (('nvidia-smi', T.NVSMI), ('python', PYSTUB),
                       ('torchrun', TORCHRUN)):
        open(osp.join(b, name), 'w').write(body)
        os.chmod(osp.join(b, name), 0o755)
    slog = osp.join(tmp, 'ql_stub.log')
    seq = itertools.count()

    def run(step, *extra, **env):
        ws = osp.join(tmp, 'ql_ws_%d' % next(seq))
        root = copy_root(tmp, osp.basename(ws))
        os.makedirs(osp.join(root, 'tools'), exist_ok=True)
        open(osp.join(root, 'tools', 'test.py'), 'w').write('')
        d = osp.join(root, 'work_dirs/diag/exec2/oracle_e0_base')
        os.makedirs(d)
        shutil.copy(fake_ref(tmp, 'r%d' % next(seq), 'X=1'),
                    osp.join(d, 'oracle_e0_base.json'))
        if osp.exists(slog):
            os.remove(slog)
        e = {'PATH': b + os.pathsep + os.environ['PATH'], 'STUB_LOG': slog,
             'REAL_PY': PY, 'CONFIG': CFG, 'CKPT': CKPT, 'CFG_OPTS': 'X=1',
             'CUDA_VISIBLE_DEVICES': '2'}
        e.update(env)
        p = subprocess.run(['bash', 'projects/QueryLoc/tools/run_queryloc.sh',
                            step] + list(extra), cwd=root, env=e,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True)
        calls = open(slog).read().splitlines() if osp.exists(slog) else []
        return p.returncode, p.stdout, calls, root

    def kinds(calls):
        return [('s0ppp' if 'queryloc_check.py s0ppp' in c else
                 '0a' if 'subcell_0a.py' in c else
                 'probe' if 'probe_step0.py' in c else
                 'st0' if 'check_load.py' in c else
                 'eval' if 'eval_ql.py' in c else
                 'pair' if 'pair_check.py' in c else
                 'rehearse' if c.startswith('torchrun') else c)
                for c in calls]

    rc, txt, calls, root = run('0a')
    assert rc == 0, txt
    assert kinds(calls) == ['s0ppp', '0a'], calls
    assert calls[1].endswith('--out-dir work_dirs/queryloc/step0a '
                             '--cfg-options X=1'), calls[1]
    assert '[gpu-check] GPU 2 idle' in txt
    logs = os.listdir(osp.join(root, 'work_dirs/queryloc/logs'))
    assert len(logs) == 1 and re.match(r'0a_\d{8}_\d{6}\.log$', logs[0])
    assert osp.isfile(osp.join(root, 'work_dirs/queryloc/s0ppp/'
                               's0ppp_check.json'))
    rc, txt, calls, _ = run('0a', CFG_OPTS='X=2')
    assert rc == 4 and kinds(calls) == ['s0ppp'], (txt, calls)
    rc, txt, calls, _ = run('0a', CUDA_VISIBLE_DEVICES='1,2')
    assert rc == 2 and 'exactly 1' in txt and kinds(calls) == ['s0ppp']
    rc, txt, calls, _ = run('0a', STUB_APPS_2='77, python, 9000 MiB')
    assert rc == 3 and kinds(calls) == ['s0ppp']
    rc, txt, calls, _ = run('0a', STUB_0A_RC='1')
    assert rc == 4 and 'self-check FAILED' in txt
    rc, txt, calls, _ = run('probe')
    assert rc == 0 and kinds(calls) == ['probe'], calls
    rc, txt, calls, _ = run('nonsense')
    assert rc == 1 and 'usage' in txt and not calls
    rc, txt, calls, root = run('0a', PROTECTED_DIRS=tmp)
    assert rc == 2 and 'protected' in txt and not calls, txt
    # step-0 entries of the v1.1 doc
    rc, txt, calls, _ = run('check-load', CUDA_VISIBLE_DEVICES='')
    assert rc == 0 and kinds(calls) == ['s0ppp', 'st0'], calls
    assert calls[1].endswith('check_load.py --ckpt %s --out-dir '
                             'work_dirs/queryloc/step0/st0 --cfg-options X=1'
                             % CKPT), calls[1]
    rc, txt, calls, _ = run('sm0')
    assert rc == 0 and kinds(calls) == ['s0ppp', 'eval'], calls
    assert re.search(r'eval_ql\.py \S+/configs/ql_mb\.py %s --tag '
                     r'sm0_mb_zero --out-dir work_dirs/queryloc/step0/sm0 '
                     r'--ref-json work_dirs/diag/exec2/oracle_e0_base/'
                     r'oracle_e0_base\.json --tol 0\.001 --debug-frames 50 '
                     r'--cfg-options X=1$' % re.escape(CKPT), calls[1]), \
        calls[1]
    rc, txt, calls, _ = run('sm0', STUB_RC='1')
    assert rc == 4 and 'S-M0 FAILED' in txt, txt
    rc, txt, calls, _ = run('sm1')
    assert rc == 0 and kinds(calls) == ['s0ppp', 'pair'] and \
        'pair_check.py sm1 --ckpt' in calls[1], calls
    rc, txt, calls, _ = run('0b')
    assert rc == 0 and kinds(calls) == ['s0ppp', 'eval'] and \
        'configs/ql_0b.py' in calls[1] and '--tag p0b_dark' in calls[1] and \
        '--tol' not in calls[1], calls
    rc, txt, calls, root = run('rehearse', 'all',
                               CUDA_VISIBLE_DEVICES='0,1,2,3')
    assert rc == 0 and kinds(calls) == ['s0ppp'] + ['rehearse'] * 4, calls
    for c, n in zip(calls[1:], ('c0', 'd3', 'mb', 'ma')):
        assert re.search(r'^torchrun --nproc_per_node=4 --master_port=29541 '
                         r'\S+/rehearse\.py \S+/configs/ql_%s\.py --ckpt '
                         r'\S+ --launcher pytorch --work-dir work_dirs/'
                         r'queryloc/step0/rehearse/ql_%s --cfg-options X=1$'
                         % (n, n), c), c
    assert txt.count('[gpu-check] GPU 3 idle') == 4, txt
    logs = os.listdir(osp.join(root, 'work_dirs/queryloc/logs'))
    assert len(logs) == 1 and logs[0].startswith('rehearse-all_'), logs
    rc, txt, calls, _ = run('rehearse', 'c0')
    assert rc == 2 and 'exactly 4' in txt and kinds(calls) == ['s0ppp']
    rc, txt, calls, _ = run('rehearse', 'xx', CUDA_VISIBLE_DEVICES='0,1,2,3')
    assert rc == 2 and 'rehearse needs one of' in txt and not calls, txt


def fn_body(text, name):
    m = re.search(r'^%s\(\) \{\n(.*?)^\}\n' % re.escape(name), text,
                  re.S | re.M)
    return m.group(1) if m else None


@case('3 run_queryloc.sh guards are verbatim copies of run_diag.sh')
def test_verbatim(tmp):
    a = open(osp.join(QD, 'run_diag.sh')).read()
    b = open(osp.join(QLT, 'run_queryloc.sh')).read()
    for name in ('inside_protected', 'need_repo_root', 'need_config',
                 'need_ckpt', 'check_gpus'):
        x, y = fn_body(a, name), fn_body(b, name)
        assert x and x == y, name
    for line in ('MEM_LIMIT_MIB=1024', 'export CUDA_DEVICE_ORDER=PCI_BUS_ID',
                 'read -r -a _opts <<< "$CFG_OPTS"',
                 'CFG_ARGS=(--cfg-options "${_opts[@]}")'):
        assert line in a and line in b, line


@case('3 code_md5.json registry matches projects/QueryLoc')
def test_registry(tmp):
    reg = json.load(open(osp.join(QL, 'code_md5.json')))
    cur = {}
    for dp, dn, fn in os.walk(QL):
        dn[:] = [d for d in dn if d != '__pycache__']
        for n in fn:
            rel = osp.relpath(osp.join(dp, n), QL)
            if rel.endswith(('.py', '.sh', '.json')) and \
                    rel != 'code_md5.json':
                cur[rel] = _md5(osp.join(dp, n))
    assert reg == cur, sorted(set(reg.items()) ^ set(cur.items()))


def main():
    global CFG, CKPT
    tmp = tempfile.mkdtemp(prefix='qd_exec3_')
    CFG = osp.join(tmp, 'fake_config.py')
    CKPT = osp.join(tmp, 'fake_ckpt.pth')
    open(CFG, 'w').write('# fake\n')
    open(CKPT, 'wb').write(os.urandom(1024))
    test_dark_exact(tmp)
    test_dark_fallback(tmp)
    test_0a_mock(tmp)
    test_s0ppp(tmp)
    test_runner(tmp)
    test_verbatim(tmp)
    test_registry(tmp)
    n_fail = sum(1 for r in T.RESULTS if r[1] != 'PASS')
    print('\n%d tests, %d failed' % (len(T.RESULTS), n_fail))
    shutil.rmtree(tmp, ignore_errors=True)
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
