#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exec doc #3 step 0 S-T0 (no GPU): build every config, compare its keys
with the start checkpoint, load it as the runner does (mmengine
load_checkpoint, strict=False) and check the initialization of new modules.

  C0, M-a  no missing, no unexpected, no size mismatch
  D3       missing == bbox_head.{decoder,prediction_heads}.{1,2}.* exactly;
           after loading, layers 1-2 equal layer 0 tensor by tensor
           (parameters and BN buffers)
  M-b      missing == bbox_head.offset_head.* exactly; after loading, the
           last offset conv (weight and bias) is all zero
  all      every checkpoint tensor present in the model is loaded exactly;
           parameter split frozen / trainable as in sec. 3 (also printed for
           S-T2)
Usage: python projects/QueryLoc/tools/check_load.py --ckpt CKPT \
           [--configs ql_c0 ql_d3 ql_mb ql_ma] [--cfg-options ...]
"""
import argparse
import os.path as osp
import re
import sys
import time

import torch

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import ql_common as qc  # noqa: E402

EXPECT_MISSING = {
    'ql_c0': None,
    'ql_ma': None,
    'ql_d3': r'^bbox_head\.(decoder|prediction_heads)\.[12]\.',
    'ql_mb': r'^bbox_head\.offset_head\.',
}


def train_ann_file(cfg):
    """Annotation file of the training set (CBGS wrapper unwrapped); exec
    docs #1-#2 only read val on the server, sm1 and the rehearsal read it."""
    ds = cfg.train_dataloader.dataset
    while 'dataset' in ds:
        ds = ds['dataset']
    path = osp.join(ds.get('data_root', '') or '', ds['ann_file'])
    return path, osp.isfile(path)


def strip_bn_count(keys):
    return {k for k in keys if not k.endswith('num_batches_tracked')}


def check_one(name, args):
    from mmengine.registry import init_default_scope
    from mmengine.runner.checkpoint import load_checkpoint

    from mmdet3d.registry import MODELS
    cfg = qc.load_cfg(osp.join(qc.QL_ROOT, 'configs', name + '.py'),
                      args.cfg_options)
    init_default_scope('mmdet3d')
    model = MODELS.build(cfg.model)
    ck = torch.load(args.ckpt, map_location='cpu')
    sd = ck.get('state_dict', ck)
    msd = model.state_dict()
    mk, ck_k = strip_bn_count(msd), strip_bn_count(sd)
    missing = sorted(mk - ck_k)
    unexpected = sorted(ck_k - mk)
    mismatch = sorted(k for k in mk & ck_k if msd[k].shape != sd[k].shape)
    pat = EXPECT_MISSING[name]
    want = sorted(k for k in mk if pat and re.match(pat, k))
    fails = []
    if missing != want:
        fails.append('missing keys differ from the new-module set '
                     '(got %d, expected %d)' % (len(missing), len(want)))
    if unexpected:
        fails.append('%d unexpected keys' % len(unexpected))
    if mismatch:
        fails.append('%d size mismatches' % len(mismatch))
    load_checkpoint(model, args.ckpt, map_location='cpu', strict=False)
    msd = model.state_dict()
    bad = [k for k in mk & ck_k if k not in mismatch and
           not torch.equal(msd[k], sd[k])]
    if bad:
        fails.append('%d tensors not loaded exactly, e.g. %s'
                     % (len(bad), bad[0]))
    init = {}
    head = model.bbox_head
    if name == 'ql_d3':
        for sub in ('decoder', 'prediction_heads'):
            s0 = getattr(head, sub)[0].state_dict()
            for i in (1, 2):
                si = getattr(head, sub)[i].state_dict()
                eq = set(s0) == set(si) and all(
                    torch.equal(s0[k], si[k]) for k in s0)
                init['%s.%d == %s.0' % (sub, i, sub)] = eq
        init['copied_layers'] = list(head.copied_layers)
    if name == 'ql_mb':
        last = head.offset_head[-1]
        init['offset_head last conv weight all zero'] = bool(
            (last.weight == 0).all())
        init['offset_head last conv bias all zero'] = bool(
            (last.bias == 0).all())
    for k, v in init.items():
        if v is False:
            fails.append('init check failed: %s' % k)
    counts = qc.param_counts(model)
    if not counts['ok']:
        fails.append('frozen / trainable split differs from sec. 3')
    return {'config': name, 'missing': missing, 'unexpected': unexpected,
            'size_mismatch': mismatch, 'expected_missing': want,
            'init_checks': init, 'params': counts,
            'result': 'FAIL' if fails else 'PASS', 'fail': fails}


def main():
    from mmengine.config import DictAction
    p = argparse.ArgumentParser(description='exec doc #3 S-T0')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--configs', nargs='+',
                   default=['ql_c0', 'ql_d3', 'ql_mb', 'ql_ma'])
    p.add_argument('--out-dir', default='work_dirs/queryloc/step0/st0')
    p.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = p.parse_args()
    t0 = time.time()
    L = ['Exec doc #3 step 0 S-T0: loading the start weights (%s)'
         % args.ckpt]
    res = {'ckpt': args.ckpt, 'ckpt_md5': qc.diag_common().file_md5(
        args.ckpt), 'runs': [], 'queryloc_md5': qc.ql_md5()}
    ann, ann_ok = train_ann_file(qc.load_cfg(
        osp.join(qc.QL_ROOT, 'configs', 'ql_c0.py'), args.cfg_options))
    res['train_ann_file'] = {'path': ann, 'exists': ann_ok}
    L.append('train set for sm1 / rehearsal: %s %s (not part of S-T0)'
             % (ann, 'found' if ann_ok else 'MISSING'))
    for name in args.configs:
        r = check_one(name, args)
        res['runs'].append(r)
        c = r['params']
        L.append('')
        L.append('%s  %s' % (name, r['result']))
        L.append('  missing %d (expected %d)  unexpected %d  size mismatch %d'
                 % (len(r['missing']), len(r['expected_missing']),
                    len(r['unexpected']), len(r['size_mismatch'])))
        for k in r['missing'][:6]:
            L.append('    missing: %s' % k)
        if len(r['missing']) > 6:
            L.append('    ... %d more' % (len(r['missing']) - 6))
        for k, v in r['init_checks'].items():
            L.append('  init: %s = %s' % (k, v))
        L.append('  params: total %d  trainable %d (expected %d)  frozen %d '
                 '(expected %d)' % (c['total'], c['trainable'],
                                    c['expected_trainable'], c['frozen'],
                                    c['expected_frozen']))
        for k, v in sorted(c['per_module'].items()):
            L.append('    %-20s total %10d  trainable %10d'
                     % (k, v['total'], v['trainable']))
        for f in r['fail']:
            L.append('  FAIL ' + f)
    ok = all(r['result'] == 'PASS' for r in res['runs'])
    res['result'] = 'PASS' if ok else 'FAIL'
    res['elapsed_s'] = round(time.time() - t0, 1)
    res['repo_version'] = qc.diag_common().read_version_line()
    L.append('')
    L.append('S-T0  %s' % res['result'])
    if not ann_ok:
        L.append('NOTE: training annotation file missing; sm1 and the '
                 'rehearsal cannot run until it is in place')
    qc.write_report(args.out_dir, 'check_load', L, res)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
