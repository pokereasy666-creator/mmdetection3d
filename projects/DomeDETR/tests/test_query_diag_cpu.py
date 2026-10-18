# -*- coding: utf-8 -*-
"""Dev-side CPU tests for tools/query_diag (not deployed to the server).

Needs torch, numpy and mmengine only (no mmcv / mmdet3d / CUDA / data).
Run from the repo root:
  python projects/DomeDETR/tests/test_query_diag_cpu.py [--ref-dir DIR]
--ref-dir: the unmodified reference implementation (query_diag.tar.gz) to
           reproduce finding F1 side by side; skipped if not given.
"""
import argparse
import glob
import json
import os
import os.path as osp
import shutil
import stat
import subprocess
import sys
import tempfile
import types

import numpy as np

REPO = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))
QD = osp.join(REPO, 'tools', 'query_diag')
MOCK = osp.join(osp.dirname(osp.abspath(__file__)), 'query_diag_mock.py')
PY = sys.executable
RESULTS = []


def case(name):
    def deco(fn):
        def run(*a, **k):
            try:
                fn(*a, **k)
                RESULTS.append((name, 'PASS', ''))
                print('[PASS] %s' % name)
            except Exception as e:  # noqa: B902
                RESULTS.append((name, 'FAIL', repr(e)))
                print('[FAIL] %s: %r' % (name, e))
        return run
    return deco


def sh(cmd, env=None, cwd=REPO):
    e = dict(os.environ)
    e.update(env or {})
    p = subprocess.run(cmd, cwd=cwd, env=e, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True)
    return p.returncode, p.stdout


def t4_near(summary):
    row = summary['T4']['0-20']         # init_ca, final_ca, init_agn, ...
    return {'init_ca': row[0], 'final_ca': row[1], 'init_agn': row[2],
            'final_agn': row[3]}


# ---------------------------------------------------------------------------

@case('recall: new impl, S1/S2/S4a/S4 on mock with verbatim in-place decode')
def test_recall_new(tmp, cfg, ckpt):
    out = osp.join(tmp, 'recall_new')
    rc, log = sh([PY, MOCK, 'recall', QD, out, cfg, ckpt])
    assert rc == 0, log[-3000:]
    s = json.load(open(osp.join(out, 'recall_summary.json')))
    assert s['init_label_match_rate'] == 1.0, s['init_label_match_rate']
    assert s['bev_pos_transposed'] is False
    assert s['center_grid_in_range_frac'] == 1.0
    t = t4_near(s)
    assert t['final_agn'] >= t['init_agn'] - 0.05, t
    assert abs(t['final_ca'] - t['init_ca']) < 0.05, t
    rt = s['runtime']
    for k in ('elapsed_s', 'versions', 'config_md5', 'checkpoint_md5',
              'fp16', 'repo_version', 'recall_loop_fp16', 'cfg_options'):
        assert k in rt, k
    assert rt['fp16'] == {'val_cfg': False, 'test_cfg': False}
    assert 'center_grid values within' in open(
        osp.join(out, 'recall_report.txt')).read()


@case('recall: reference impl reproduces F1 (final recall collapses)')
def test_recall_ref(tmp, cfg, ckpt, ref_dir):
    out = osp.join(tmp, 'recall_ref')
    rc, log = sh([PY, MOCK, 'recall', ref_dir, out, cfg, ckpt])
    assert rc == 0, log[-3000:]
    s = json.load(open(osp.join(out, 'recall_summary.json')))
    assert s['init_label_match_rate'] == 1.0
    t = t4_near(s)
    assert t['final_agn'] < t['init_agn'] - 0.05, t     # S4 would FAIL
    assert t['final_agn'] < 0.05, t
    print('       ref T4 0-20m: init_agn=%.3f final_agn=%.3f'
          % (t['init_agn'], t['final_agn']))


@case('merge: S checks PASS on new run; rule inputs match direct recompute')
def test_merge_new(tmp):
    d = osp.join(tmp, 'recall_new')
    rc, log = sh([PY, osp.join(QD, 'diag_merge_bins.py'), d])
    assert rc == 0, log[-3000:]
    m = json.load(open(osp.join(d, 'recall_merged.json')))
    assert m['self_checks_all_pass'] is True
    assert [c['id'] for c in m['self_checks']] == \
        ['S1', 'S2', 'S3a', 'S3b', 'S4a', 'S4']
    assert 'R1' not in json.dumps(m['rule_inputs'])
    Z = np.load(osp.join(d, 'recall_per_gt.npz'))
    ti = list(Z['thr']).index(2.0)
    far = Z['dist'] >= 30
    near = Z['dist'] < 30
    r200 = float(np.mean(Z['rank_ca'][far, ti] < 200))
    r1000 = float(np.mean(Z['rank_ca'][far, ti] < 1000))
    agn200 = float(np.mean(Z['rank_agn'][far, ti] < 200))
    fin = float(np.mean(Z['fin_ca'][far, ti]))
    rn200 = float(np.mean(Z['rank_ca'][near, ti] < 200))
    ri = m['rule_inputs']
    assert abs(ri['R_far(200)'] - r200) < 1e-12
    assert abs(ri['R_far(1000)'] - r1000) < 1e-12
    assert abs(ri['R_near(200)'] - rn200) < 1e-12
    assert abs(ri['dR_far = R_far(1000)-R_far(200)'] - (r1000 - r200)) < 1e-12
    assert abs(ri['agn_gap_far = agnR_far(200)-R_far(200)'] -
               (agn200 - r200)) < 1e-12
    assert abs(ri['final_gap_far = final_ca_far-R_far(200)'] -
               (fin - r200)) < 1e-12
    assert rn200 < 1.0, 'mock should leave some near misses'
    assert abs(ri['miss_far/miss_near'] -
               (1 - r200) / (1 - rn200)) < 1e-9
    assert ri['n_gt_near'] == int(near.sum())
    assert ri['n_gt_far'] == int(far.sum())
    fc = Z['frame_counts']
    assert abs(ri['frac_over_200'] - float(np.mean(fc > 200))) < 1e-12
    print('       far: R(200)=%.3f R(1000)=%.3f agn(200)=%.3f final=%.3f '
          'N=%d' % (r200, r1000, agn200, fin, int(far.sum())))


@case('merge: reference-run output fails S2/S4a (missing) and S4')
def test_merge_ref(tmp):
    d = osp.join(tmp, 'recall_ref')
    rc, log = sh([PY, osp.join(QD, 'diag_merge_bins.py'), d])
    assert rc == 1, log[-2000:]
    m = json.load(open(osp.join(d, 'recall_merged.json')))
    res = {c['id']: c['result'] for c in m['self_checks']}
    assert res['S1'] == 'PASS' and res['S4'] == 'FAIL', res
    assert res['S2'] == 'MISSING' and res['S4a'] == 'MISSING', res


def run_oracle(tmp, cfg, ckpt, tag, *extra):
    wd = osp.join(tmp, 'oracle', 'oracle_' + tag)
    rc, log = sh([PY, MOCK, 'oracle', QD, wd, cfg, ckpt, tag] + list(extra))
    assert rc == 0, log[-3000:]
    r = json.load(open(osp.join(wd, 'oracle_%s.json' % tag)))
    h = json.load(open(osp.join(wd, 'harness_checks_%s.json' % tag)))
    return r, h


@case('oracle: base / loc / loc_cls / far -> S6, S7, no confidence leak')
def test_oracle(tmp, cfg, ckpt):
    base, _ = run_oracle(tmp, cfg, ckpt, 'base', '--oracle', 'none')
    assert base['stats']['injected_cells'] == 0
    assert 'runtime' in base and base['runtime']['checkpoint_md5']
    pick = lambda r: r['metrics'][  # noqa: E731
        'NuScenes metric/pred_instances_3d_NuScenes/mAP']
    for tag, extra in (('all', ['--oracle', 'loc', '--range', '0', '1000']),
                       ('all_cls', ['--oracle', 'loc_cls', '--range', '0',
                                    '1000']),
                       ('far', ['--oracle', 'loc', '--range', '30', '1000'])):
        r, h = run_oracle(tmp, cfg, ckpt, tag, *extra)
        st = r['stats']
        assert st['injected_cells'] > 0, st
        assert st['init_mismatch_rate'] <= 0.01, st          # S6
        assert st['capped_cells'] == 0, st
        assert st['restored_queries'] >= 0.99 * st['injected_cells'], st
        assert h['restored_bad'] == 0, h                     # no leak
        assert h['injected_selected'] == st['injected_cells'], (h, st)
        assert pick(r) >= pick(base), (pick(r), pick(base))
        print('       %-7s injected=%d restored=%d proxy=%.3f (base %.3f)'
              % (tag, st['injected_cells'], st['restored_queries'],
                 pick(r), pick(base)))


@case('summary: S5-S7 lines printed, no R verdicts')
def test_summary(tmp):
    root = osp.join(tmp, 'oracle')
    gate = {'check': 'S5', 'result': 'PASS', 'tol': 0.001,
            'diff': {'mAP': 0.0, 'NDS': 0.0}}
    json.dump(gate, open(osp.join(root, 'oracle_base', 's5_gate.json'), 'w'))
    rc, log = sh([PY, osp.join(QD, 'diag_summarize.py'), root])
    assert rc == 0, log
    assert 'S5  base vs known' in log and log.count('  S6  ') == 3 and \
        log.count('  S7  ') == 3, log
    assert 'FAIL' not in log, log
    saved = open(osp.join(root, 'summary.txt')).read()
    assert saved.splitlines()[0].startswith('tag') and 'S7  far' in saved
    assert '[summary] saved' not in saved


def write_base(path, m, n):
    os.makedirs(osp.dirname(path), exist_ok=True)
    json.dump({'args': {'oracle': 'none', 'tag': 'base'},
               'metrics': {'x/mAP': m, 'x/NDS': n}, 'stats': {}},
              open(path, 'w'))


@case('S5 gate: tolerance edges incl. binary float slack')
def test_gate(tmp):
    g = osp.join(QD, 'diag_gate_s5.py')
    b = osp.join(tmp, 'gate', 'oracle_base.json')
    write_base(b, 0.6660, 0.6970)
    for known, want in (((0.6650, 0.6960), 0), ((0.6670, 0.6980), 0),
                        ((0.6649, 0.6970), 1), ((0.6660, 0.6981), 1)):
        rc, log = sh([PY, g, 'gate', '--base-json', b, '--map',
                      str(known[0]), '--nds', str(known[1]), '--out',
                      osp.join(tmp, 'gate', 'g.json')])
        assert rc == want, (known, rc, log)
    assert 'not run the other four' not in log or rc == 1


@case('S5 adjudication: SOURCE / HOOK verdicts, json and text-log sources')
def test_adjudicate(tmp):
    g = osp.join(QD, 'diag_gate_s5.py')
    b = osp.join(tmp, 'adj', 'oracle_base.json')
    write_base(b, 0.66512, 0.69601)
    nh = osp.join(tmp, 'adj', 'nohook', '20261017_000000')
    os.makedirs(nh)
    json.dump({'NuScenes metric/pred_instances_3d_NuScenes/mAP': 0.66530,
               'NuScenes metric/pred_instances_3d_NuScenes/NDS': 0.69590},
              open(osp.join(nh, '20261017_000000.json'), 'w'))
    rc, log = sh([PY, g, 'adjudicate', '--base-json', b, '--nohook-dir',
                  osp.dirname(nh), '--map', '0.6700', '--nds', '0.7000'])
    assert rc == 0 and 'verdict SOURCE' in log and 'CANDIDATE' in log, log
    json.dump({'x/mAP': 0.66580, 'x/NDS': 0.69601},
              open(osp.join(nh, '20261017_000000.json'), 'w'))
    rc, log = sh([PY, g, 'adjudicate', '--base-json', b, '--nohook-dir',
                  osp.dirname(nh)])
    assert rc == 1 and 'verdict HOOK' in log, log
    os.remove(osp.join(nh, '20261017_000000.json'))
    open(osp.join(nh, '20261017_000000.log'), 'w').write(
        'Epoch(test) [6019/6019] NuScenes metric/pred_instances_3d_NuScenes/'
        'mAP: 0.6651  NuScenes metric/pred_instances_3d_NuScenes/NDS: '
        '0.6960\n')
    rc, log = sh([PY, g, 'adjudicate', '--base-json', b, '--nohook-dir',
                  osp.dirname(nh)])
    assert rc == 0 and 'text log' in log, log


# ---------------------------------------------------------------------------
# run_diag.sh with stubbed nvidia-smi / python
# ---------------------------------------------------------------------------

NVSMI = r'''#!/usr/bin/env bash
id=""; q=""
while [ $# -gt 0 ]; do
  case "$1" in
    -i) id=$2; shift 2 ;;
    --query-compute-apps=*) q=apps; shift ;;
    --query-gpu=*) q=mem; shift ;;
    *) shift ;;
  esac
done
v="STUB_APPS_$id"; m="STUB_MEM_$id"
if [ "$q" = apps ]; then printf '%s' "${!v:-}"; else echo "${!m:-3}"; fi
'''

PYSTUB = r'''#!/usr/bin/env bash
echo "FORCE_CUDA=${FORCE_CUDA:-} python $*" >> "$STUB_LOG"
if [[ "$*" == *setup.py*build_ext*--inplace* ]]; then
  touch projects/BEVFusion/bevfusion/ops/bev_pool/bev_pool_ext.cpython-38-x86_64-linux-gnu.so
  touch projects/BEVFusion/bevfusion/ops/voxel/voxel_layer.cpython-38-x86_64-linux-gnu.so
  exit 0
fi
case "$*" in
  *diag_gate_s5.py*|*diag_merge_bins.py*) exec "$REAL_PY" "$@" ;;
esac
if [[ "$*" == *diag_oracle_query.py* ]]; then
  wd=""; tag=""; mode=none
  args=("$@")
  for ((i=0; i<${#args[@]}; i++)); do
    case "${args[$i]}" in
      --work-dir) wd=${args[$((i+1))]} ;;
      --tag) tag=${args[$((i+1))]} ;;
      --oracle) mode=${args[$((i+1))]} ;;
    esac
  done
  mkdir -p "$wd"
  echo "{\"args\": {\"oracle\": \"$mode\", \"tag\": \"$tag\", \"range\": [0, 1000], \"score\": \"orig\"}, \"metrics\": {\"x/mAP\": $STUB_MAP, \"x/NDS\": $STUB_NDS}, \"stats\": {\"injected_cells\": 0}}" > "$wd/oracle_$tag.json"
fi
if [[ "$*" == *tools/test.py* ]]; then
  args=("$@")
  for ((i=0; i<${#args[@]}; i++)); do
    [ "${args[$i]}" = --work-dir ] && wd=${args[$((i+1))]}
  done
  mkdir -p "$wd/20261017_000000"
  echo "{\"x/mAP\": $STUB_MAP, \"x/NDS\": $STUB_NDS}" > "$wd/20261017_000000/20261017_000000.json"
fi
exit 0
'''


def stub_env(tmp):
    b = osp.join(tmp, 'bin')
    os.makedirs(b, exist_ok=True)
    for name, body in (('nvidia-smi', NVSMI), ('python', PYSTUB)):
        p = osp.join(b, name)
        open(p, 'w').write(body)
        os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC)
    return b


@case('run_diag.sh: C3 CONFIG required, C5 GPU checks, C2 S5 gating')
def test_run_diag(tmp, cfg, ckpt):
    b = stub_env(tmp)
    log = osp.join(tmp, 'stub.log')
    out = osp.join(tmp, 'rd_out')
    base = {'PATH': b + os.pathsep + os.environ['PATH'], 'STUB_LOG': log,
            'REAL_PY': PY, 'OUT': out, 'STUB_MAP': '0.6651',
            'STUB_NDS': '0.6962'}
    base.pop('CUDA_VISIBLE_DEVICES', None)
    rdsh = osp.join(QD, 'run_diag.sh')

    def run(step, **env):
        e = dict(base)
        e.update(env)
        if os.path.exists(log):
            os.remove(log)
        rc, txt = sh(['bash', rdsh, step], env=e)
        calls = [c.split(' ', 1)[1] for c in open(log).read().splitlines()] \
            if osp.exists(log) else []
        return rc, txt, calls

    clean = {'CUDA_VISIBLE_DEVICES': ''}
    rc, txt, calls = run('quick', **clean)
    assert rc == 2 and 'CONFIG' in txt and not calls, (rc, txt)
    rc, txt, calls = run('quick', CONFIG=cfg, CKPT=ckpt, **clean)
    assert rc == 2 and 'CUDA_VISIBLE_DEVICES' in txt and not calls, txt
    rc, txt, calls = run('quick', CONFIG=cfg, CKPT=ckpt,
                         CUDA_VISIBLE_DEVICES='2', CFG_OPTS='a.b=1 c=x')
    assert rc == 0 and len(calls) == 1, (rc, txt, calls)
    assert '--max-samples 300' in calls[0] and \
        calls[0].endswith('--cfg-options a.b=1 c=x'), calls
    assert 'GPU 2 idle' in txt, txt
    logs = sorted(glob.glob(osp.join(out, 'logs', 'quick_*.log')))
    assert logs and 'GPU 2 idle' in open(logs[-1]).read(), logs
    rc, txt, calls = run('recall', CONFIG=cfg, CKPT=ckpt,
                         CUDA_VISIBLE_DEVICES='2',
                         STUB_APPS_2='4242, python, 9000 MiB')
    assert rc == 3 and not calls and 'compute processes' in txt, txt
    rc, txt, calls = run('recall', CONFIG=cfg, CKPT=ckpt,
                         CUDA_VISIBLE_DEVICES='2', STUB_MEM_2='2048')
    assert rc == 3 and not calls and '1024' in txt, txt
    rc, txt, calls = run('recall', CONFIG=cfg, CKPT=ckpt,
                         CUDA_VISIBLE_DEVICES='2', STUB_MEM_2='1024')
    assert rc == 0 and len(calls) == 1, txt              # == limit is OK
    rc, txt, calls = run('recall', CONFIG=cfg, CKPT=ckpt,
                         CUDA_VISIBLE_DEVICES='1,2')
    assert rc == 2 and not calls and 'exactly 1' in txt, txt
    rc, txt, calls = run('oracle', CONFIG=cfg, CKPT=ckpt,
                         CUDA_VISIBLE_DEVICES='2')
    assert rc == 2 and 'BASE_MAP' in txt and not calls, txt
    rc, txt, calls = run('oracle', CONFIG=cfg, CKPT=ckpt,
                         CUDA_VISIBLE_DEVICES='2', BASE_MAP='0.6700',
                         BASE_NDS='0.6962')
    tags = [c.split('--tag ')[1].split()[0] for c in calls
            if 'diag_oracle_query.py' in c]
    assert rc == 4 and tags == ['base'], (rc, tags, txt)
    assert 'NOT run' in txt, txt
    rc, txt, calls = run('oracle', CONFIG=cfg, CKPT=ckpt,
                         CUDA_VISIBLE_DEVICES='2', BASE_MAP='0.6651',
                         BASE_NDS='0.6962')
    tags = [c.split('--tag ')[1].split()[0] for c in calls
            if 'diag_oracle_query.py' in c]
    assert rc == 0 and tags == ['base', 'all', 'far', 'near', 'all_cls'], \
        (rc, tags, txt)
    assert osp.isfile(osp.join(out, 'oracle_base', 's5_gate.json'))
    rc, txt, calls = run('oracle', CONFIG=cfg, CKPT=ckpt, GPUS='3',
                         CUDA_VISIBLE_DEVICES='0,1,2', BASE_MAP='0.6651',
                         BASE_NDS='0.6962')
    assert rc == 2 and not calls, txt
    rc, txt, calls = run('s5-adjudicate', CONFIG=cfg, CKPT=ckpt,
                         CUDA_VISIBLE_DEVICES='2', STUB_MAP='0.6653')
    assert rc == 4 and 'verdict SOURCE' in txt, txt
    assert any('tools/test.py' in c for c in calls), calls
    rc, txt, calls = run('merge')
    assert rc == 2 and 'no recall_per_gt.npz' in txt, txt
    rc, txt, calls = run('s0', CONFIG=cfg, CKPT=ckpt,
                         TRAIN_WORK_DIR='work_dirs/baseline1',
                         BASE_MAP='0.6648', BASE_NDS='0.7060')
    assert rc == 0 and 'diag_s0_check.py' in calls[0] and \
        '--train-work-dir work_dirs/baseline1' in calls[0] and \
        '--base-map 0.6648 --base-nds 0.7060' in calls[0], calls


# ---------------------------------------------------------------------------
# S0 checker pieces that run without mmcv / mmdet3d
# ---------------------------------------------------------------------------

def load_s0():
    sys.path.insert(0, QD)
    import diag_s0_check as s0
    s0.LINES[:] = []
    return s0


def stub_project_modules(root):
    for name, rel in (('mmdet3d', 'mmdet3d/__init__.py'),
                      ('projects', 'projects'),
                      ('projects.BEVFusion', 'projects/BEVFusion'),
                      ('projects.BEVFusion.bevfusion',
                       'projects/BEVFusion/bevfusion/__init__.py')):
        m = types.ModuleType(name)
        m.__file__ = osp.join(root, rel)
        m.__path__ = []
        sys.modules[name] = m
    sys.modules['projects'].BEVFusion = sys.modules['projects.BEVFusion']
    sys.modules['projects.BEVFusion'].bevfusion = \
        sys.modules['projects.BEVFusion.bevfusion']


RUN_NAME = 'bevfusion_lidar-cam_official6e_4xa30_amp512_accum4_seed577127641'


def snapshot(root):
    out = {}
    for dp, dn, fn in os.walk(root):
        for n in dn + fn:
            p = osp.join(dp, n)
            st = os.lstat(p)
            out[osp.relpath(p, root)] = (st.st_mtime_ns, st.st_size)
    return out


def make_fake_ep(tmp):
    """Fake EP-Fusion repo + a nuScenes dir outside it."""
    ep = osp.join(tmp, 'ep_repo')
    run = osp.join(ep, 'work_dirs', RUN_NAME)
    os.makedirs(osp.join(run, '20260718_020825', 'vis_data'))
    os.makedirs(osp.join(ep, 'checkpoints'))
    os.makedirs(osp.join(ep, 'data'))
    open(osp.join(run, 'epoch_5.pth'), 'wb').write(os.urandom(4096))
    open(osp.join(run, 'epoch_6.pth'), 'wb').write(os.urandom(64))
    open(osp.join(run, 'bevfusion_lidar-cam_voxel0075_4xa30-amp-accum_'
                  'nus-3d.py'), 'w').write('x = 1\n')
    open(osp.join(run, '20260718_020825', '20260718_020825.log'),
         'w').write('log\n')
    open(osp.join(run, '20260718_020825', 'vis_data', 'scalars.json'),
         'w').write('{"x/mAP": 0.6648, "x/NDS": 0.706, "step": 5}\n')
    open(osp.join(ep, 'checkpoints', 'swint-nuimages-pretrained.pth'),
         'wb').write(os.urandom(512))
    nus = osp.join(tmp, 'datasets', 'nuscenes')
    os.makedirs(nus)
    open(osp.join(nus, 'nuscenes_infos_val.pkl'), 'wb').write(b'pkl')
    os.symlink(nus, osp.join(ep, 'data', 'nuscenes'))
    return ep, nus


def make_workspace(tmp, name):
    ws = osp.join(tmp, name)
    for d in ('projects/BEVFusion/bevfusion/ops/bev_pool',
              'projects/BEVFusion/bevfusion/ops/voxel', 'tools/query_diag'):
        os.makedirs(osp.join(ws, d))
    open(osp.join(ws, 'tools', 'test.py'), 'w').write('')
    return ws


@case('A-1 fetch: read-only copy, md5 manifest, nuScenes link, guards')
def test_fetch(tmp):
    ep, nus = make_fake_ep(tmp)
    ws = make_workspace(tmp, 'ws_fetch')
    fetch = osp.join(QD, 'exec01_fetch_inputs.sh')
    before = snapshot(ep)
    rc, out = sh(['bash', fetch], env={'EP_REPO': ep}, cwd=ws)
    assert rc == 0, out
    assert snapshot(ep) == before, 'EP repo was modified'
    inp = osp.join(ws, 'work_dirs', 'exec01_inputs')
    for rel in ('epoch_5.pth', 'swint-nuimages-pretrained.pth',
                'train_run/bevfusion_lidar-cam_voxel0075_4xa30-amp-accum_'
                'nus-3d.py',
                'train_run/20260718_020825/vis_data/scalars.json',
                'train_run/20260718_020825/20260718_020825.log'):
        assert osp.isfile(osp.join(inp, rel)), rel
    assert not osp.exists(osp.join(inp, 'epoch_6.pth'))
    man = open(osp.join(inp, 'FETCH_MANIFEST.txt')).read()
    assert man.count(' -> ') == 5, man
    assert os.readlink(osp.join(ws, 'data', 'nuscenes')) == nus
    rc, out = sh(['bash', fetch], env={'EP_REPO': ep}, cwd=ep)
    assert rc == 2 and not osp.exists(osp.join(ep, 'work_dirs',
                                               'exec01_inputs')), out
    rc, out = sh(['bash', fetch], env={'EP_REPO': ep,
                                       'INPUTS': osp.join(ep, 'x')}, cwd=ws)
    assert rc == 2 and 'inside the EP-Fusion repo' in out, out
    assert snapshot(ep) == before


@case('exec01_env.sh: workspace-local inputs; refuses in EP repo / elsewhere')
def test_env(tmp):
    ep = osp.join(tmp, 'ep_repo')                 # made by test_fetch
    ws = osp.join(tmp, 'ws_fetch')
    env = osp.join(QD, 'exec01_env.sh')
    show = ('source "%s" && echo "CKPT=$CKPT" && echo "TW=$TRAIN_WORK_DIR" '
            '&& echo "OPTS=$CFG_OPTS" && echo "B=$BASE_MAP/$BASE_NDS"' % env)
    rc, out = sh(['bash', '-c', show], env={'EP_REPO': ep}, cwd=ws)
    inp = osp.join(ws, 'work_dirs', 'exec01_inputs')
    assert rc == 0, out
    assert 'CKPT=%s/epoch_5.pth' % inp in out, out
    assert 'TW=%s/train_run' % inp in out, out
    assert 'OPTS=model.img_backbone.init_cfg.checkpoint=%s/swint-nuimages-' \
        'pretrained.pth' % inp in out, out
    assert 'B=0.6648/0.7060' in out, out
    assert 'WARNING: missing %s' % 'projects/BEVFusion/configs' in out, out
    rc, out = sh(['bash', '-c', 'source "%s"; echo "rc=$?"' % env],
                 env={'EP_REPO': ep}, cwd=osp.join(ep, 'work_dirs'))
    assert 'rc=1' in out and 'inside the EP-Fusion repo' in out, out
    rc, out = sh(['bash', '-c', 'source "%s"; echo "rc=$?"' % env],
                 env={'EP_REPO': ep}, cwd=tmp)
    assert 'rc=1' in out and 'workspace root' in out, out


@case('run_diag.sh A-1 guards; build-ops is in-place with FORCE_CUDA=1')
def test_guards(tmp):
    b = stub_env(tmp)
    ep = osp.join(tmp, 'ep_repo')
    ws = make_workspace(tmp, 'ws_build')
    log = osp.join(tmp, 'guard.log')
    rdsh = osp.join(QD, 'run_diag.sh')
    env = {'PATH': b + os.pathsep + os.environ['PATH'], 'STUB_LOG': log,
           'REAL_PY': PY, 'PROTECTED_DIRS': ep}
    before = snapshot(ep)
    rc, out = sh(['bash', rdsh, 'merge'], env=env, cwd=osp.join(ep, 'data'))
    assert rc == 2 and 'protected' in out, out
    e = dict(env, OUT=osp.join(ep, 'work_dirs', 'diag'))
    rc, out = sh(['bash', rdsh, 'summary'], env=e, cwd=ws)
    assert rc == 2 and 'OUT=' in out, out
    e = dict(env, OUT=osp.join(ws, 'work_dirs', 'diag'))
    rc, out = sh(['bash', rdsh, 'build-ops'], env=e, cwd=ws)
    assert rc == 0, out
    calls = open(log).read().splitlines()
    assert calls == ['FORCE_CUDA=1 python projects/BEVFusion/setup.py '
                     'build_ext --inplace'], calls
    assert 'develop' not in open(rdsh).read().split('build-ops)')[1] \
        .split(';;')[0].replace("'setup.py develop'", '')
    e = dict(env, OUT=osp.join(ws, 'work_dirs', 'diag'), CONFIG=__file__,
             CKPT=osp.join(ep, 'work_dirs', RUN_NAME, 'epoch_5.pth'))
    rc, out = sh(['bash', rdsh, 'merge'], env=e, cwd=ws)
    assert 'NOTE: CKPT is read (read-only) from a protected' in out, out
    assert snapshot(ep) == before


@case('S0a: this repo matches the 82c2b4e manifest')
def test_s0a_repo():
    s0 = load_s0()
    stub_project_modules(REPO)
    r = s0.s0a(REPO)
    assert r['result'] == 'PASS', r['fail']
    assert all(x['status'] == 'OK' for x in r['files'])
    assert all(x['status'] == 'OK' for x in r['extended'])


@case('S0a: detects DIFF / EXTRA / outside-repo import; .md is WARN only')
def test_s0a_tamper(tmp):
    s0 = load_s0()
    fake = osp.join(tmp, 'fake_repo')
    man = json.load(open(s0.MANIFEST))
    for p in list(man['tree']) + list(man['extended']):
        os.makedirs(osp.dirname(osp.join(fake, p)), exist_ok=True)
        shutil.copy(osp.join(REPO, p), osp.join(fake, p))
    stub_project_modules(fake)
    assert s0.s0a(fake)['result'] == 'PASS'
    with open(osp.join(fake, 'projects/BEVFusion/README.md'), 'a') as f:
        f.write('\nedit\n')
    r = s0.s0a(fake)
    assert r['result'] == 'PASS' and any('README.md' in w for w in r['warn'])
    with open(osp.join(fake, 'projects/BEVFusion/bevfusion/utils.py'),
              'a') as f:
        f.write('\n# local edit\n')
    open(osp.join(fake, 'projects/BEVFusion/bevfusion/ep_module.py'),
         'w').write('x = 1\n')
    os.makedirs(osp.join(fake, 'projects/BEVFusion/bevfusion/ops/voxel/'
                         '__pycache__'), exist_ok=True)
    open(osp.join(fake, 'projects/BEVFusion/bevfusion/ops/voxel/'
                  'voxel_layer.cpython-38-x86_64-linux-gnu.so'),
         'w').write('bin')
    r = s0.s0a(fake)
    assert r['result'] == 'FAIL'
    joined = '\n'.join(r['fail'])
    assert 'DIFF projects/BEVFusion/bevfusion/utils.py (core)' in joined
    d = r['core_diffs']['projects/BEVFusion/bevfusion/utils.py']
    assert '+# local edit' in d and d.startswith('--- 82c2b4e:'), d[:300]
    assert 'EXTRA projects/BEVFusion/bevfusion/ep_module.py' in joined
    assert '.so' not in joined
    shutil.copy(osp.join(REPO, 'projects/BEVFusion/bevfusion/utils.py'),
                osp.join(fake, 'projects/BEVFusion/bevfusion/utils.py'))
    os.remove(osp.join(fake, 'projects/BEVFusion/bevfusion/ep_module.py'))
    stub_project_modules(REPO)                 # imports resolve elsewhere
    r = s0.s0a(fake)
    assert r['result'] == 'FAIL' and 'IMPORT' in '\n'.join(r['fail'])


def load_cfg(path):
    from mmengine.config import Config
    return Config.fromfile(osp.join(REPO, path), import_custom_modules=False)


def make_fake_repo(tmp, name):
    s0 = load_s0()
    fake = osp.join(tmp, name)
    man = json.load(open(s0.MANIFEST))
    for p in list(man['tree']) + list(man['extended']):
        os.makedirs(osp.dirname(osp.join(fake, p)), exist_ok=True)
        shutil.copy(osp.join(REPO, p), osp.join(fake, p))
    stub_project_modules(fake)
    return s0, fake


@case('S0a rule D-3: July-run style config edit accepted; eval edits fail')
def test_s0a_rule_d3(tmp):
    s0, fake = make_fake_repo(tmp, 'fake_repo_d3')
    cfg4 = osp.join(fake, 'projects/BEVFusion/configs/'
                    'bevfusion_lidar-cam_voxel0075_4xa30-amp-accum_nus-3d.py')
    lidar = osp.join(fake, 'projects/BEVFusion/configs/'
                     'bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-'
                     '20e_nus-3d.py')
    orig4, orig_l = open(cfg4).read(), open(lidar).read()
    july = (orig4.replace('max_epochs=20', 'max_epochs=6')
            .replace('max_epoch=20', 'max_epoch=6')
            .replace('T_max=20', 'T_max=6')
            .replace('/212022085500129/mmdetection3d-claude-jolly',
                     '/212022085500129/bevfusion-concat/mmdetection3d-'
                     'claude-jolly'))
    assert july != orig4
    open(cfg4, 'w').write(july)
    r = s0.s0a(fake)
    assert r['result'] == 'PASS', r['fail']
    assert r['config_diff_accepted'] == [osp.relpath(cfg4, fake)]
    fd = r['config_field_diffs'][osp.relpath(cfg4, fake)]
    paths = sorted(d['path'] for d in fd)
    assert 'model.img_backbone.init_cfg.checkpoint' in paths, paths
    assert 'train_cfg.max_epochs' in paths and all(d['allowed'] for d in fd)
    assert r['core_diffs'], 'unified diff expected for a changed config'
    for extra, bad in (("\ntest_cfg = dict(fp16=True)\n", 'test_cfg.fp16'),
                       ("\nmodel = dict(bbox_head=dict(test_cfg=dict("
                        "nms_type='circle')))\n",
                        'model.bbox_head.test_cfg.nms_type'),
                       ("\ndefault_hooks = dict(checkpoint=dict("
                        "max_keep_ckpts=-1))\n",
                        'default_hooks.checkpoint.max_keep_ckpts')):
        open(cfg4, 'w').write(july + extra)
        r = s0.s0a(fake)
        assert r['result'] == 'FAIL', (bad, r)
        fd = r['config_field_diffs'][osp.relpath(cfg4, fake)]
        assert [d['path'] for d in fd if not d['allowed']] == [bad], fd
    open(cfg4, 'w').write(orig4 + '\n# comment only\n')
    r = s0.s0a(fake)
    assert r['result'] == 'PASS' and r['config_diff_accepted'], r['fail']
    assert r['config_field_diffs'][osp.relpath(cfg4, fake)] == []
    open(cfg4, 'w').write(orig4)
    open(lidar, 'w').write(orig_l.replace('num_proposals=200',
                                          'num_proposals=300'))
    r = s0.s0a(fake)
    assert r['result'] == 'FAIL'
    assert any('bevfusion_lidar_voxel0075' in m for m in r['fail']), r['fail']
    open(lidar, 'w').write(orig_l)
    assert s0.s0a(fake)['result'] == 'PASS'


@case('S0c helper: checkpoint composition by top-level module')
def test_s0c_modules():
    s0 = load_s0()
    keys = ['img_backbone.a', 'img_backbone.b', 'pts_backbone.c',
            'bbox_head.heatmap_head.0.conv.weight', 'fusion_layer.0.weight']
    c = s0.module_counts(keys)
    assert c == {'img_backbone': 2, 'pts_backbone': 1, 'bbox_head': 1,
                 'fusion_layer': 1}, c
    lidar_only = s0.module_counts(['pts_backbone.c', 'bbox_head.x'])
    assert not [m for m in s0.CAMERA_MODULES if lidar_only.get(m)]


@case('S0b: override list of 4xa30 vs official; no new model types')
def test_s0b_diff():
    s0 = load_s0()
    off = s0.plain(load_cfg(s0.OFFICIAL_FUSION).to_dict())
    cur = s0.plain(load_cfg('projects/BEVFusion/configs/'
                            'bevfusion_lidar-cam_voxel0075_4xa30-amp-accum_'
                            'nus-3d.py').to_dict())
    diffs = s0.tree_diff(off, cur)
    paths = [p for p, _, _ in diffs]
    print('       %d overrides:' % len(diffs))
    for p, a, b in diffs:
        print('         %s: %s -> %s' % (p, s0.short(a, 50), s0.short(b, 50)))
    assert 'model.img_backbone.init_cfg.checkpoint' in paths
    assert [p for p in paths if p.startswith('model.')] == \
        ['model.img_backbone.init_cfg.checkpoint'], paths
    assert 'train_cfg.max_epochs' in paths
    assert not any(p.startswith(('test_', 'val_')) for p in paths), paths
    t_off = s0.collect_types(off['model'])
    t_cur = s0.collect_types(cur['model'])
    assert t_off == t_cur and 'TransFusionHead' in t_cur
    assert 'Pretrained' not in t_cur                 # init_cfg skipped
    assert off['custom_imports'] == cur['custom_imports']


@case('S0d: eval-setup comparison, pretty_text round trip, val metrics')
def test_s0d(tmp):
    s0 = load_s0()
    from mmengine.config import Config
    path = ('projects/BEVFusion/configs/'
            'bevfusion_lidar-cam_voxel0075_4xa30-amp-accum_nus-3d.py')
    cfg = load_cfg(path)
    cur = s0.plain(cfg.to_dict())
    assert s0.tree_diff(cur['val_dataloader']['dataset'],
                        cur['test_dataloader']['dataset']) == []
    assert s0.tree_diff(cur['val_evaluator'], cur['test_evaluator']) == []
    # The checkpoint stores cfg.pretty_text; it must parse back to the same
    # evaluation setup (custom_imports stubbed: no mmcv here).
    stub_project_modules(REPO)
    back = s0.plain(Config.fromstring(cfg.pretty_text, '.py').to_dict())
    assert s0.compare_eval_setup('rt', back, cur) == []
    tr = json.loads(json.dumps(back))
    tr['sync_bn'] = 'torch'
    tr['model']['img_backbone']['init_cfg']['checkpoint'] = '/other/swin.pth'
    tr['model']['bbox_head']['train_cfg']['gaussian_overlap'] = 0.2
    assert s0.compare_eval_setup('rt', tr, cur) == []      # train-only keys
    tr['model']['bbox_head']['test_cfg']['nms_type'] = 'circle'
    tr['val_cfg'] = {'fp16': True}
    tr['val_dataloader']['dataset']['ann_file'] = 'other.pkl'
    out = s0.compare_eval_setup('rt', tr, cur)
    assert len(out) == 3, out
    wd = osp.join(tmp, 'baseline1', '20260601_000000', 'vis_data')
    os.makedirs(wd)
    with open(osp.join(wd, 'scalars.json'), 'w') as f:
        f.write(json.dumps({'loss': 1.0, 'step': 100}) + '\n')
        f.write(json.dumps({'NuScenes metric/pred_instances_3d_NuScenes/'
                            'mAP': 0.66512, 'NuScenes metric/'
                            'pred_instances_3d_NuScenes/NDS': 0.69601,
                            'step': 19}) + '\n')
    rows = s0.read_val_metrics(osp.join(tmp, 'baseline1'))
    assert len(rows) == 1 and rows[0]['step'] == 19 and \
        abs(rows[0]['mAP'] - 0.66512) < 1e-12, rows
    ck = {'meta': {'epoch': 19, 'cfg': cfg.pretty_text}}
    twd = osp.join(tmp, 'baseline1')
    r = s0.s0d(cfg, path, ck, twd, 0.6651, 0.6960)
    assert r['result'] == 'PASS', r['fail']
    assert r['base_check']['log'] == ('0.6651', '0.6960')
    assert r['ckpt_meta_run']['ckpt_epoch'] == 19
    r = s0.s0d(cfg, path, ck, twd, 0.6650, 0.6960)
    assert r['result'] == 'FAIL' and 'do not match' in r['fail'][0], r
    r = s0.s0d(cfg, path, {'meta': {'epoch': 5, 'cfg': cfg.pretty_text}},
               twd, 0.6651, 0.6960)
    assert r['result'] == 'FAIL' and 'no val metrics' in r['fail'][0], r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref-dir', default=None)
    ap.add_argument('--keep', action='store_true')
    args = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix='qd_test_')
    cfg = osp.join(tmp, 'fake_config.py')
    ckpt = osp.join(tmp, 'fake_ckpt.pth')
    open(cfg, 'w').write('# fake\n')
    open(ckpt, 'wb').write(os.urandom(1024))
    test_recall_new(tmp, cfg, ckpt)
    if args.ref_dir:
        test_recall_ref(tmp, cfg, ckpt, args.ref_dir)
        test_merge_ref(tmp)
    test_merge_new(tmp)
    test_oracle(tmp, cfg, ckpt)
    test_summary(tmp)
    test_gate(tmp)
    test_adjudicate(tmp)
    test_run_diag(tmp, cfg, ckpt)
    test_fetch(tmp)
    test_env(tmp)
    test_guards(tmp)
    test_s0a_repo()
    test_s0a_tamper(tmp)
    test_s0a_rule_d3(tmp)
    test_s0c_modules()
    test_s0b_diff()
    test_s0d(tmp)
    n_fail = sum(1 for r in RESULTS if r[1] != 'PASS')
    print('\n%d tests, %d failed' % (len(RESULTS), n_fail))
    if args.keep:
        print('kept', tmp)
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
