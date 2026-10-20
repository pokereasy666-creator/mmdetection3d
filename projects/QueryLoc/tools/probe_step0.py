#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exec doc #3 step 0: read-only server probe (no GPU work, no writes except
the report). Answers what the dev side cannot read locally:

  T4  mmdet gaussian_focal_loss source (positive = target == 1 ?)
  T5  optimizer / schedule / batch of the run that produced the start
      weights (the config dumped into the training work dir copy)
  4.2 GPU inventory (memory, compute processes) and free disk space, for
      choosing the number of GPUs of the fine-tuning protocol
Writes <out-dir>/probe_step0_<time>.txt (single report, CLAUDE.md rule 18).
Usage: python projects/QueryLoc/tools/probe_step0.py \
           [--train-run work_dirs/exec01_inputs/train_run]
"""
import argparse
import glob
import hashlib
import inspect
import os
import os.path as osp
import platform
import subprocess
import sys
import time

L = []


def out(s=''):
    L.append(str(s))
    print(s)


def read_version_line(path='VERSION'):
    try:
        with open(path) as f:
            return f.readline().strip() or 'unknown'
    except (OSError, UnicodeDecodeError):
        return 'unknown'


def sh(cmd):
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=60)
        return r.stdout.decode(errors='replace').rstrip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return 'unavailable: %r' % (e, )


def section(title):
    out('')
    out('== %s ==' % title)


def versions():
    section('versions')
    out('python %s' % platform.python_version())
    for name in ('torch', 'mmcv', 'mmengine', 'mmdet', 'mmdet3d', 'numpy'):
        try:
            mod = __import__(name)
            out('%-9s %s  (%s)' % (name, getattr(mod, '__version__', '?'),
                                   osp.dirname(mod.__file__)))
        except Exception as e:  # noqa: B902
            out('%-9s import failed: %r' % (name, e))


def t4():
    section('T4 mmdet gaussian_focal_loss (positive definition)')
    try:
        from mmdet.models.losses import gaussian_focal_loss as g
    except Exception as e:  # noqa: B902
        out('import failed: %r' % (e, ))
        return
    for name in ('gaussian_focal_loss', 'gaussian_focal_loss_with_pos_inds'):
        fn = getattr(g, name, None)
        if fn is None:
            continue
        src, start = inspect.getsourcelines(fn)
        out('%s  %s:%d-%d' % (name, inspect.getsourcefile(fn), start,
                              start + len(src) - 1))
        for i, line in enumerate(src):
            out('  %4d  %s' % (start + i, line.rstrip()))
    src = inspect.getsource(g.gaussian_focal_loss)
    out('T4 verdict: positives are %s' % (
        '`gaussian_target.eq(1)` (as assumed)' if 'eq(1)' in src
        else 'NOT defined by eq(1) -- check the source above'))
    try:
        import mmdet3d.models.utils.gaussian as gm
        p = inspect.getsourcefile(gm)
        out('mmdet3d draw_heatmap_gaussian: %s md5 %s' % (
            p, hashlib.md5(open(p, 'rb').read()).hexdigest()))
    except Exception as e:  # noqa: B902
        out('mmdet3d gaussian import failed: %r' % (e, ))


def t5(train_run):
    section('T5 config of the run that produced the start weights (%s)'
            % train_run)
    cfgs = sorted(glob.glob(osp.join(train_run, '*.py')))
    if not cfgs:
        out('no dumped config (*.py) under %s' % train_run)
        return
    try:
        from mmengine.config import Config
    except Exception as e:  # noqa: B902
        out('mmengine import failed: %r' % (e, ))
        return
    for path in cfgs:
        out('file %s md5 %s' % (path, hashlib.md5(
            open(path, 'rb').read()).hexdigest()))
        try:
            cfg = Config.fromfile(path, import_custom_modules=False)
        except Exception as e:  # noqa: B902
            out('  Config.fromfile failed: %r' % (e, ))
            continue
        for key in ('optim_wrapper', 'param_scheduler', 'train_cfg',
                    'auto_scale_lr', 'randomness', 'load_from', 'sync_bn',
                    'find_unused_parameters', 'model_wrapper_cfg'):
            out('  %s = %r' % (key, cfg.get(key, '<absent>')))
        td = cfg.get('train_dataloader', {})
        out('  train_dataloader.batch_size = %r, num_workers = %r, '
            'dataset.type = %r' % (td.get('batch_size'),
                                   td.get('num_workers'),
                                   td.get('dataset', {}).get('type')))
    logs = sorted(glob.glob(osp.join(train_run, '*', '*.log')))
    for lp in logs[:1]:
        section('T5 launch lines of %s' % lp)
        with open(lp, errors='replace') as f:
            for i, line in enumerate(f):
                if i > 400:
                    break
                if any(k in line for k in ('sync_bn', 'launcher',
                                           'Distributed', 'GPU 0',
                                           'world_size', 'seed')):
                    out('  ' + line.rstrip()[:200])


def gpus():
    section('4.2 GPU inventory (read-only nvidia-smi)')
    out(sh(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,'
            'utilization.gpu', '--format=csv']))
    out(sh(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,'
            'used_memory', '--format=csv']))
    out(sh(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader']))
    section('disk space (work_dirs)')
    out(sh(['df', '-h', 'work_dirs' if osp.isdir('work_dirs') else '.']))


def main():
    p = argparse.ArgumentParser(description='exec doc #3 step 0 probe')
    p.add_argument('--out-dir', default='work_dirs/queryloc')
    p.add_argument('--train-run', default='work_dirs/exec01_inputs/train_run')
    args = p.parse_args()
    out('Exec doc #3 step 0 probe  %s  repo_version: %s  cwd: %s'
        % (time.strftime('%Y-%m-%d %H:%M:%S'), read_version_line(),
           os.getcwd()))
    versions()
    t4()
    t5(args.train_run)
    gpus()
    os.makedirs(args.out_dir, exist_ok=True)
    path = osp.join(args.out_dir, 'probe_step0_%s.txt'
                    % time.strftime('%Y%m%d_%H%M%S'))
    with open(path, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('[probe] saved %s' % path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
