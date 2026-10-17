#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S0a-S0d: server consistency checks before step 1 (exec doc v1.1 sec. 5).

Run from the SERVER repo root, CPU only, before tools/query_diag/ is copied
into that repo (the script can run from an unpacked branch zip):

  cd <server repo root>
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD python <zip>/tools/query_diag/\
diag_s0_check.py --config <CONFIG> --ckpt <CKPT> \
      [--train-work-dir work_dirs/baseline1] [--cfg-options ...]

S0a  md5 of the audited files against the manifest of commit 82c2b4e,
     no extra source files under projects/BEVFusion, and mmdet3d /
     projects.BEVFusion imported from this repo. No git is used (the
     server copy comes from a zip).
S0b  every override of the 4xa30 config relative to the official fusion
     config; the model must not introduce non-official module types
     (type-set comparison + registry provenance of every model type).
S0c  build the model on CPU and load CKPT through mmengine's
     load_checkpoint (the call Runner.load_checkpoint makes), capturing its
     log: no missing keys / unexpected keys / size mismatch.
S0d  test vs val dataloader and evaluator identical; fp16 of val_cfg /
     test_cfg recorded; the training config embedded in CKPT
     (meta['cfg']) must match the current evaluation setup. With
     --train-work-dir the dumped config is compared as well and the val
     metrics of every epoch are listed (source of BASE_MAP / BASE_NDS).

Writes <out-dir>/s0_report.txt and <out-dir>/s0_report.json.
Exit code 0 only if S0a-S0d are all PASS.
"""
import argparse
import copy
import difflib
import glob
import hashlib
import json
import logging
import os
import os.path as osp
import re
import shutil
import sys
import tempfile
import time
import traceback

HERE = osp.dirname(osp.abspath(__file__))
MANIFEST = osp.join(HERE, 's0_manifest_82c2b4e.json')
OFFICIAL_FUSION = ('projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_'
                   'second_secfpn_8xb4-cyclic-20e_nus-3d.py')
SOURCE_EXT = ('.py', '.c', '.cc', '.cpp', '.cu', '.cuh', '.h', '.hpp')
IGNORE_DIRS = ('__pycache__', 'build', '.eggs')
IGNORE_EXT = ('.pyc', '.pyo', '.so', '.o')
ALLOWED_MODULE_PREFIX = ('projects.BEVFusion.', 'mmdet3d.', 'mmdet.',
                         'mmcv.', 'mmengine.', 'torch.')
CAMERA_MODULES = ('img_backbone', 'img_neck', 'view_transform',
                  'fusion_layer')
DIFF_LINES_SHOWN = 80
# Pre-registered rule D-3 (projects/DomeDETR/docs/EXEC01_DECISIONS.md,
# 2026-10-17): a core config file whose md5 differs is accepted only if
# every field-level difference of its resolved config lies under one of
# these paths, AND S0d passes. The four core .py files stay strict.
CONFIG_DIFF_ALLOWED = (
    'param_scheduler', 'train_cfg',              # schedule
    'optim_wrapper', 'auto_scale_lr',            # optimizer
    'train_dataloader',                          # batch (+ its dataset)
    'train_pipeline',                            # training pipeline
    'model.img_backbone.init_cfg.checkpoint',    # Swin init path
)
RUNTIME_BASE = 'configs/_base_/default_runtime.py'

LINES = []


def emit(s=''):
    LINES.append(s)
    print(s)


def short(v, n=160):
    s = repr(v)
    return s if len(s) <= n else s[:n - 3] + '...'


def file_md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def plain(x):
    """ConfigDict / tuple -> plain dict / list, recursively."""
    if hasattr(x, 'items'):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    return x


def tree_diff(a, b, path=''):
    """List of (path, a_value, b_value) where the two trees differ."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            p = '%s.%s' % (path, k) if path else str(k)
            if k not in a:
                out.append((p, '<absent>', b[k]))
            elif k not in b:
                out.append((p, a[k], '<absent>'))
            else:
                out += tree_diff(a[k], b[k], p)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            out += tree_diff(x, y, '%s[%d]' % (path, i))
    elif a != b:
        out.append((path, a, b))
    return out


def drop_keys(tree, names):
    """Copy of tree without any dict entry whose key is in names."""
    if isinstance(tree, dict):
        return {k: drop_keys(v, names) for k, v in tree.items()
                if k not in names}
    if isinstance(tree, list):
        return [drop_keys(v, names) for v in tree]
    return tree


def collect_types(tree, skip=('init_cfg', )):
    out = set()
    if isinstance(tree, dict):
        for k, v in tree.items():
            if k in skip:
                continue
            if k == 'type' and isinstance(v, str):
                out.add(v)
            else:
                out |= collect_types(v, skip)
    elif isinstance(tree, list):
        for v in tree:
            out |= collect_types(v, skip)
    return out


def inside(path, root):
    path, root = osp.realpath(path), osp.realpath(root)
    return path == root or path.startswith(root + os.sep)


# ---------------------------------------------------------------------------
# S0a
# ---------------------------------------------------------------------------

def diff_allowed(path):
    return any(path == a or path.startswith((a + '.', a + '['))
               for a in CONFIG_DIFF_ALLOWED)


def config_field_diffs(repo, man):
    """Field-level diffs of every core config, audited vs server.

    The audited texts are written into a temp tree mirroring the repo
    layout, with the SERVER's default_runtime.py as the common base on both
    sides, so only changes inside the three config files show up. No
    project code is imported (import_custom_modules=False).
    """
    from mmengine.config import Config
    cfgs = sorted(p for p in man['core'] if '/configs/' in p)
    tmp = tempfile.mkdtemp(prefix='s0a_cfg_')
    out = {}
    try:
        for p in cfgs:
            dst = osp.join(tmp, p)
            os.makedirs(osp.dirname(dst), exist_ok=True)
            with open(dst, 'w') as f:
                f.write(man['core_text'][p])
        dst = osp.join(tmp, RUNTIME_BASE)
        os.makedirs(osp.dirname(dst), exist_ok=True)
        shutil.copy(osp.join(repo, RUNTIME_BASE), dst)
        for p in cfgs:
            a = plain(Config.fromfile(osp.join(tmp, p),
                                      import_custom_modules=False).to_dict())
            b = plain(Config.fromfile(osp.join(repo, p),
                                      import_custom_modules=False).to_dict())
            out[p] = [{'path': q, 'audited': x, 'server': y,
                       'allowed': diff_allowed(q)}
                      for q, x, y in tree_diff(a, b)]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def s0a(repo):
    with open(MANIFEST) as f:
        man = json.load(f)
    core = set(man['core'])
    res = {'commit': man['commit'], 'files': [], 'extra': [],
           'extended': [], 'imports': {}, 'core_diffs': {},
           'config_diff_accepted': []}
    fail, warn, cfg_diff = [], [], []
    for p, ref in sorted(man['tree'].items()):
        fp = osp.join(repo, p)
        if not osp.isfile(fp):
            st = 'MISSING'
        else:
            st = 'OK' if file_md5(fp) == ref else 'DIFF'
        if st == 'DIFF' and p in man.get('core_text', {}):
            with open(fp, errors='replace') as f:
                server = f.read()
            res['core_diffs'][p] = ''.join(difflib.unified_diff(
                man['core_text'][p].splitlines(True),
                server.splitlines(True), fromfile='82c2b4e:' + p,
                tofile='server:' + p, n=1))
        is_source = p.endswith(SOURCE_EXT) or p in core
        level = 'core' if p in core else ('source' if is_source else 'doc')
        res['files'].append({'path': p, 'status': st, 'level': level})
        if st == 'DIFF' and p in core and '/configs/' in p:
            cfg_diff.append(p)          # decided by rule D-3 below
        elif st != 'OK':
            (fail if is_source else warn).append('%s %s (%s)' % (st, p,
                                                                 level))
    known = set(man['tree'])
    base = osp.join(repo, 'projects/BEVFusion')
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs
                   if d not in IGNORE_DIRS and not d.endswith('.egg-info')]
        for fn in files:
            rel = osp.relpath(osp.join(root, fn), repo).replace(os.sep, '/')
            if rel in known or fn.endswith(IGNORE_EXT):
                continue
            src = fn.endswith(SOURCE_EXT)
            res['extra'].append({'path': rel, 'source': src})
            if src:
                fail.append('EXTRA %s (untracked source file)' % rel)
    if cfg_diff:
        try:
            fd = config_field_diffs(repo, man)
        except Exception:  # noqa: B902
            fd = None
            fail += ['DIFF %s (core config)' % p for p in cfg_diff]
            fail.append('field-level config comparison failed:\n' +
                        traceback.format_exc())
        if fd is not None:
            res['config_field_diffs'] = fd
            bad = sorted(p for p, v in fd.items()
                         if any(not d['allowed'] for d in v))
            for p in bad:
                fail.append('config %s has field changes outside rule D-3'
                            % p)
            if not bad:
                res['config_diff_accepted'] = list(cfg_diff)
                for p in cfg_diff:
                    warn.append('DIFF %s accepted under rule D-3 (%d field '
                                'changes, all allowed); valid only if S0d '
                                'PASS' % (p, len(fd.get(p, []))))
    for p, ref in sorted(man['extended'].items()):
        fp = osp.join(repo, p)
        if not osp.isfile(fp):
            st = 'MISSING'
        else:
            st = 'OK' if file_md5(fp) == ref else 'DIFF'
        res['extended'].append({'path': p, 'status': st})
        if st != 'OK':
            warn.append('%s %s (extended set, not part of the S0a '
                        'criterion)' % (st, p))
    if not fail:
        # Only import project code once the files are known to match.
        try:
            import mmdet3d
            import projects.BEVFusion.bevfusion as bf
            res['imports'] = {'mmdet3d': osp.realpath(mmdet3d.__file__),
                              'projects.BEVFusion.bevfusion':
                              osp.realpath(bf.__file__)}
            for name, p in res['imports'].items():
                if not inside(p, repo):
                    fail.append('IMPORT %s resolves outside the repo: %s'
                                % (name, p))
        except Exception:  # noqa: B902
            fail.append('IMPORT failed:\n' + traceback.format_exc())
    res['fail'], res['warn'] = fail, warn
    res['result'] = 'FAIL' if fail else 'PASS'
    n_ok = sum(1 for x in res['files'] if x['status'] == 'OK')
    emit('S0a  md5 vs manifest of %s: %d/%d tracked files OK, %d extra '
         'files, extended set %d/%d OK'
         % (man['commit'][:7], n_ok, len(res['files']), len(res['extra']),
            sum(1 for x in res['extended'] if x['status'] == 'OK'),
            len(res['extended'])))
    for x in res['files']:
        if x['level'] == 'core':
            emit('     core %-8s %s' % (x['status'], x['path']))
    for name, p in res['imports'].items():
        emit('     import %s -> %s' % (name, p))
    for m in fail:
        emit('     FAIL ' + m)
    for m in warn:
        emit('     WARN ' + m)
    for p, v in sorted(res.get('config_field_diffs', {}).items()):
        if v:
            emit('     field changes of %s (resolved config):' % p)
        for d in v:
            emit('       %-11s %s: %s -> %s'
                 % ('allowed' if d['allowed'] else 'NOT ALLOWED', d['path'],
                    short(d['audited'], 60), short(d['server'], 60)))
    for p, d in res['core_diffs'].items():
        lines = d.splitlines()
        emit('     diff of %s (%d lines%s):'
             % (p, len(lines), ', first %d shown, full text in json'
                % DIFF_LINES_SHOWN if len(lines) > DIFF_LINES_SHOWN else ''))
        for ln in lines[:DIFF_LINES_SHOWN]:
            emit('       ' + ln)
    emit('S0a  %s' % res['result'])
    return res


# ---------------------------------------------------------------------------
# S0b
# ---------------------------------------------------------------------------

def resolve_type(t):
    from mmdet3d.registry import MODELS, TASK_UTILS
    for reg in (MODELS, TASK_UTILS):
        try:
            cls = reg.get(t)
        except Exception:  # noqa: B902
            cls = None
        if cls is not None:
            return cls
    return None


def s0b(repo, config):
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    off = Config.fromfile(osp.join(repo, OFFICIAL_FUSION))
    cur = Config.fromfile(config)
    init_default_scope(cur.get('default_scope', 'mmdet3d'))
    d_off, d_cur = plain(off.to_dict()), plain(cur.to_dict())
    diffs = tree_diff(d_off, d_cur)
    fail = []
    t_off = collect_types(d_off.get('model', {}))
    t_cur = collect_types(d_cur.get('model', {}))
    new_types = sorted(t_cur - t_off)
    if new_types:
        fail.append('model introduces types absent from the official '
                    'config: %s' % new_types)
    if d_off.get('custom_imports') != d_cur.get('custom_imports'):
        fail.append('custom_imports differ: official=%s current=%s'
                    % (d_off.get('custom_imports'),
                       d_cur.get('custom_imports')))
    prov, unresolved = {}, []
    for t in sorted(t_off | t_cur):
        cls = resolve_type(t)
        if cls is None:
            unresolved.append(t)
            continue
        mod = cls.__module__
        try:
            import inspect
            path = osp.realpath(inspect.getfile(cls))
        except (TypeError, OSError):
            path = '?'
        prov[t] = {'module': mod, 'file': path}
        if not mod.startswith(ALLOWED_MODULE_PREFIX):
            fail.append('type %s registered by non-official module %s (%s)'
                        % (t, mod, path))
        elif mod.startswith(('projects.', 'mmdet3d.')) and \
                not inside(path, repo):
            fail.append('type %s loaded from outside the repo: %s'
                        % (t, path))
    res = {'official': OFFICIAL_FUSION, 'current': config,
           'overrides': [{'path': p, 'official': a, 'current': b}
                         for p, a, b in diffs],
           'model_types_official': sorted(t_off),
           'model_types_current': sorted(t_cur),
           'new_model_types': new_types, 'provenance': prov,
           'unresolved_types': unresolved, 'fail': fail,
           'result': 'FAIL' if fail else 'PASS'}
    emit('S0b  overrides of %s relative to %s: %d'
         % (osp.basename(config), osp.basename(OFFICIAL_FUSION), len(diffs)))
    for p, a, b in diffs:
        emit('     %s: %s -> %s' % (p, short(a, 70), short(b, 70)))
    emit('     model types: official %d, current %d, new %s'
         % (len(t_off), len(t_cur), new_types or 'none'))
    if unresolved:
        emit('     WARN types not found in MODELS / TASK_UTILS: %s'
             % unresolved)
    for m in fail:
        emit('     FAIL ' + m)
    emit('S0b  %s' % res['result'])
    return res


# ---------------------------------------------------------------------------
# S0c
# ---------------------------------------------------------------------------

def module_counts(keys):
    out = {}
    for k in keys:
        top = k.split('.', 1)[0]
        out[top] = out.get(top, 0) + 1
    return out


class _ListHandler(logging.Handler):

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


def s0c(cfg, ckpt):
    from mmengine.registry import init_default_scope
    from mmengine.runner.checkpoint import load_checkpoint

    from mmdet3d.registry import MODELS
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    res = {'checkpoint': ckpt}
    ck = None
    try:
        model = MODELS.build(copy.deepcopy(cfg.model))
        log = logging.getLogger('query_diag.s0c')
        log.setLevel(logging.DEBUG)
        log.propagate = False
        h = _ListHandler()
        log.addHandler(h)
        # Same call chain as Runner.load_checkpoint (strict=False).
        ck = load_checkpoint(model, ckpt, map_location='cpu', strict=False,
                             logger=log, revise_keys=[(r'^module.', '')])
        log.removeHandler(h)
        text = '\n'.join(h.records)
        low = text.lower()
        hits = {'missing keys': 'missing keys' in low,
                'unexpected keys': 'unexpected key' in low,
                'size mismatch': 'size mismatch' in low}
        sd = ck.get('state_dict', ck) if isinstance(ck, dict) else {}
        sd = {re.sub(r'^module\.', '', k): v for k, v in sd.items()}
        msd = model.state_dict()
        raw = {
            'missing': [k for k in msd if k not in sd
                        and 'num_batches_tracked' not in k],
            'unexpected': [k for k in sd if k not in msd],
            'shape': ['%s ckpt%s model%s' % (k, tuple(sd[k].shape),
                                             tuple(msd[k].shape))
                      for k in msd if k in sd and hasattr(sd[k], 'shape')
                      and tuple(sd[k].shape) != tuple(msd[k].shape)]}
        meta = ck.get('meta', {}) if isinstance(ck, dict) else {}
        ck_mod, md_mod = module_counts(sd), module_counts(msd)
        cam_ck = [m for m in CAMERA_MODULES if ck_mod.get(m)]
        cam_md = [m for m in CAMERA_MODULES if md_mod.get(m)]
        modality_fail = bool(cam_md) and not cam_ck
        res.update({
            'ckpt_modules': ck_mod, 'model_modules': md_mod,
            'ckpt_has_camera_branch': bool(cam_ck),
            'model_has_camera_branch': bool(cam_md),
            'modality_mismatch': modality_fail,
            'load_log': text, 'log_hits': hits,
            'raw_diff_counts': {k: len(v) for k, v in raw.items()},
            'raw_diff_examples': {k: v[:10] for k, v in raw.items()},
            'n_model_tensors': len(msd), 'n_ckpt_tensors': len(sd),
            'meta': {k: meta.get(k) for k in ('epoch', 'iter',
                                              'mmengine_version', 'time',
                                              'experiment_name', 'seed')},
            'result': 'FAIL' if any(hits.values()) or modality_fail
            else 'PASS'})
    except Exception:  # noqa: B902
        res.update({'error': traceback.format_exc(), 'result': 'FAIL'})
    emit('S0c  load %s (CPU, mmengine load_checkpoint, strict=False)' % ckpt)
    if 'error' in res:
        emit('     FAIL exception while building / loading:')
        for ln in res['error'].rstrip().splitlines():
            emit('       ' + ln)
        emit('     hint: if img_backbone.init_cfg.checkpoint does not exist '
             'here, pass CFG_OPTS="model.img_backbone.init_cfg.checkpoint='
             '<local swin ckpt>"')
    else:
        emit('     ckpt tensors by module: %s' % res['ckpt_modules'])
        emit('     camera branch: ckpt=%s, CONFIG model=%s'
             % (res['ckpt_has_camera_branch'],
                res['model_has_camera_branch']))
        if res['modality_mismatch']:
            emit('     FAIL CONFIG is lidar-cam but CKPT has no camera branch '
                 '(LiDAR-only checkpoint?)')
        emit('     log hits: %s' % res['log_hits'])
        emit('     raw key/shape diff before load hooks (info): %s'
             % res['raw_diff_counts'])
        emit('     tensors: model %d, ckpt %d; ckpt meta: %s'
             % (res['n_model_tensors'], res['n_ckpt_tensors'], res['meta']))
        if res['load_log']:
            emit('     load log:')
            for ln in res['load_log'].splitlines()[:40]:
                emit('       ' + ln)
    emit('S0c  %s' % res['result'])
    return res, ck


# ---------------------------------------------------------------------------
# S0d
# ---------------------------------------------------------------------------

EVAL_KEYS_DROPPED_FROM_MODEL = ('init_cfg', 'train_cfg')


def compare_eval_setup(tag, train_cfg_dict, cur):
    """Training-time val setup vs the current test setup."""
    out = []
    pairs = [
        ('val_dataloader.dataset', 'test_dataloader.dataset',
         train_cfg_dict.get('val_dataloader', {}).get('dataset'),
         cur.get('test_dataloader', {}).get('dataset')),
        ('val_evaluator', 'test_evaluator',
         train_cfg_dict.get('val_evaluator'), cur.get('test_evaluator')),
        ('model (no init_cfg/train_cfg)', 'model (no init_cfg/train_cfg)',
         drop_keys(train_cfg_dict.get('model'), EVAL_KEYS_DROPPED_FROM_MODEL),
         drop_keys(cur.get('model'), EVAL_KEYS_DROPPED_FROM_MODEL)),
    ]
    for a_name, b_name, a, b in pairs:
        for p, x, y in tree_diff(a, b):
            out.append('%s: %s[%s]=%s vs current %s[%s]=%s'
                       % (tag, a_name, p, short(x, 60), b_name, p,
                          short(y, 60)))
    tf = bool((train_cfg_dict.get('val_cfg') or {}).get('fp16', False))
    cf = bool((cur.get('test_cfg') or {}).get('fp16', False))
    if tf != cf:
        out.append('%s: val_cfg.fp16=%s vs current test_cfg.fp16=%s'
                   % (tag, tf, cf))
    return out


def read_val_metrics(work_dir):
    rows = []
    for p in sorted(glob.glob(osp.join(work_dir, '*', 'vis_data',
                                       'scalars.json'))):
        with open(p) as f:
            for ln in f:
                try:
                    d = json.loads(ln)
                except ValueError:
                    continue
                m = [v for k, v in d.items() if k.endswith('mAP')]
                n = [v for k, v in d.items() if k.endswith('NDS')]
                if m and n:
                    rows.append({'log': p, 'step': d.get('step'),
                                 'mAP': float(m[0]), 'NDS': float(n[0])})
    return rows


def s0d(cfg, config, ck, train_work_dir, base_map=None, base_nds=None):
    from mmengine.config import Config
    cur = plain(cfg.to_dict())
    fail, info = [], []
    v_ds = cur.get('val_dataloader', {}).get('dataset')
    t_ds = cur.get('test_dataloader', {}).get('dataset')
    for p, a, b in tree_diff(v_ds, t_ds):
        fail.append('val vs test dataloader.dataset differ at %s: %s vs %s'
                    % (p, short(a, 60), short(b, 60)))
    for p, a, b in tree_diff(cur.get('val_evaluator'),
                             cur.get('test_evaluator')):
        fail.append('val_evaluator vs test_evaluator differ at %s: %s vs %s'
                    % (p, short(a, 60), short(b, 60)))
    fp16 = {'val_cfg': bool((cur.get('val_cfg') or {}).get('fp16', False)),
            'test_cfg': bool((cur.get('test_cfg') or {}).get('fp16',
                                                             False))}
    if fp16['val_cfg'] != fp16['test_cfg']:
        fail.append('val_cfg.fp16 != test_cfg.fp16: %s' % fp16)
    res = {'ann_file': {'val': (v_ds or {}).get('ann_file'),
                        'test': (t_ds or {}).get('ann_file')},
           'data_root': {'val': (v_ds or {}).get('data_root'),
                         'test': (t_ds or {}).get('data_root')},
           'fp16': fp16}

    sources = 0
    meta = ck.get('meta', {}) if isinstance(ck, dict) else {}
    if meta.get('cfg'):
        try:
            tcfg = plain(Config.fromstring(meta['cfg'], '.py').to_dict())
            fail += compare_eval_setup('ckpt meta cfg', tcfg, cur)
            res['ckpt_meta_sync_bn'] = tcfg.get('sync_bn')
            res['ckpt_meta_optim_wrapper_type'] = \
                (tcfg.get('optim_wrapper') or {}).get('type')
            res['ckpt_meta_run'] = {
                'work_dir': tcfg.get('work_dir'),
                'load_from': tcfg.get('load_from'),
                'input_modality': tcfg.get('input_modality'),
                'max_epochs': (tcfg.get('train_cfg') or {}).get('max_epochs'),
                'ckpt_epoch': meta.get('epoch')}
            sources += 1
        except Exception:  # noqa: B902
            fail.append('cannot parse ckpt meta cfg:\n' +
                        traceback.format_exc())
    else:
        info.append('CKPT has no meta.cfg (not saved by mmengine Runner?)')
    if train_work_dir:
        dumps = sorted(glob.glob(osp.join(train_work_dir, '*.py')))
        want = osp.basename(config)
        pick = [p for p in dumps if osp.basename(p) == want] or \
            (dumps if len(dumps) == 1 else [])
        if pick:
            res['train_dump'] = pick[0]
            try:
                dcfg = plain(Config.fromfile(pick[0]).to_dict())
                fail += compare_eval_setup('train dump', dcfg, cur)
                sources += 1
            except Exception:  # noqa: B902
                fail.append('cannot parse %s:\n%s'
                            % (pick[0], traceback.format_exc()))
        else:
            info.append('no unique dumped config in %s (found %s)'
                        % (train_work_dir, dumps))
        res['val_metrics'] = read_val_metrics(train_work_dir)
        # BASE values must equal the CKPT epoch's val metrics as mmengine
        # prints them ('%.4f'), i.e. come from this very evaluation.
        ep = meta.get('epoch')
        rows = [r for r in res['val_metrics'] if r['step'] == ep]
        if base_map is not None and base_nds is not None:
            if not rows:
                fail.append('no val metrics for CKPT epoch %s under %s; '
                            'cannot verify BASE_MAP / BASE_NDS'
                            % (ep, train_work_dir))
            else:
                r = rows[-1]
                got = ('%.4f' % r['mAP'], '%.4f' % r['NDS'])
                want = ('%.4f' % base_map, '%.4f' % base_nds)
                res['base_check'] = {'log': got, 'given': want,
                                     'epoch': ep, 'source': r['log']}
                if got != want:
                    fail.append('BASE_MAP/BASE_NDS %s do not match the '
                                'epoch-%s val log %s' % (want, ep, got))
    elif base_map is not None:
        info.append('BASE values not cross-checked (no --train-work-dir)')
    res['training_config_sources_compared'] = sources
    res['fail'], res['info'] = fail, info
    if fail:
        res['result'] = 'FAIL'
    elif sources == 0:
        res['result'] = 'INCOMPLETE'
    else:
        res['result'] = 'PASS'

    emit('S0d  evaluation setup')
    emit('     ann_file val=%s test=%s' % (res['ann_file']['val'],
                                         res['ann_file']['test']))
    emit('     fp16: %s' % fp16)
    if 'ckpt_meta_sync_bn' in res:
        emit('     training run (ckpt meta): sync_bn=%s optim_wrapper=%s '
             '(train-only; SyncBN in eval mode equals BN)'
             % (res['ckpt_meta_sync_bn'],
                res['ckpt_meta_optim_wrapper_type']))
    if 'ckpt_meta_run' in res:
        for k, v in res['ckpt_meta_run'].items():
            emit('     training run (ckpt meta) %s: %s' % (k, v))
    emit('     training configs compared: %d (ckpt meta cfg%s)'
         % (sources, ', train dump %s' % res['train_dump']
            if 'train_dump' in res else ''))
    ep = meta.get('epoch')
    for r in res.get('val_metrics', []):
        mark = '  <- CKPT epoch' if ep is not None and r['step'] == ep else ''
        emit('     val step %-3s mAP=%.6f NDS=%.6f  (%s)%s'
             % (r['step'], r['mAP'], r['NDS'], r['log'], mark))
    if 'base_check' in res:
        emit('     BASE check (epoch %s): given mAP/NDS %s, log %s'
             % (res['base_check']['epoch'], res['base_check']['given'],
                res['base_check']['log']))
    for m in info:
        emit('     INFO ' + m)
    for m in fail:
        emit('     FAIL ' + m)
    emit('S0d  %s' % res['result'])
    return res


# ---------------------------------------------------------------------------

def main():
    from mmengine.config import DictAction
    p = argparse.ArgumentParser(description='S0a-S0d consistency checks')
    p.add_argument('--config', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out-dir', default='work_dirs/diag/s0')
    p.add_argument('--train-work-dir', default=None,
                   help='training work dir of CKPT (dumped config + logs)')
    p.add_argument('--repo', default='.', help='repo root (default: cwd)')
    p.add_argument('--base-map', type=float, default=None,
                   help='BASE_MAP to verify against the training val log')
    p.add_argument('--base-nds', type=float, default=None,
                   help='BASE_NDS to verify against the training val log')
    p.add_argument('--cfg-options', nargs='+', action=DictAction,
                   help='same CFG_OPTS as the later runs (key=value ...)')
    args = p.parse_args()
    repo = osp.abspath(args.repo)
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    out = {'repo': repo, 'config': args.config, 'ckpt': args.ckpt,
           'config_md5': file_md5(args.config),
           'ckpt_md5': file_md5(args.ckpt),
           'cfg_options': args.cfg_options, 'cfg_opts_env':
           os.environ.get('CFG_OPTS', ''),
           'time': time.strftime('%Y-%m-%d %H:%M:%S')}
    emit('[s0] repo=%s config=%s ckpt=%s' % (repo, args.config, args.ckpt))
    emit('[s0] config md5=%s ckpt md5=%s' % (out['config_md5'],
                                            out['ckpt_md5']))
    out['S0a'] = s0a(repo)
    if out['S0a']['result'] != 'PASS':
        emit('[s0] S0a failed: S0b-S0d skipped (project code not imported).')
    else:
        from mmengine.config import Config
        out['S0b'] = s0b(repo, args.config)
        cfg = Config.fromfile(args.config)
        if args.cfg_options:
            cfg.merge_from_dict(args.cfg_options)
            emit('[s0] NOTE: cfg_options applied for S0c/S0d: %s'
                 % args.cfg_options)
        out['S0c'], ck = s0c(cfg, args.ckpt)
        out['S0d'] = s0d(cfg, args.config, ck, args.train_work_dir,
                         args.base_map, args.base_nds)
        del ck
    results = {k: out[k]['result'] for k in ('S0a', 'S0b', 'S0c', 'S0d')
               if k in out}
    if out['S0a'].get('config_diff_accepted'):
        emit('[s0] S0a accepted config diffs under rule D-3 (%s); this holds '
             'only together with S0d PASS (S0d=%s).'
             % (', '.join(osp.basename(p) for p in
                          out['S0a']['config_diff_accepted']),
                results.get('S0d', 'not run')))
    ok = len(results) == 4 and all(v == 'PASS' for v in results.values())
    out['results'] = results
    out['all_pass'] = ok
    out['elapsed_s'] = round(time.time() - t0, 1)
    emit('')
    emit('[s0] results: %s -> %s' % (results, 'ALL PASS' if ok else
                                     'NOT PASSED (stop and report)'))
    with open(osp.join(args.out_dir, 's0_report.txt'), 'w') as f:
        f.write('\n'.join(LINES) + '\n')
    with open(osp.join(args.out_dir, 's0_report.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print('[s0] saved %s/s0_report.{txt,json}' % args.out_dir)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
