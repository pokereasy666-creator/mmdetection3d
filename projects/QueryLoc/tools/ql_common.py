# -*- coding: utf-8 -*-
"""Shared helpers of the exec doc #3 server tools (no GPU work here)."""
import importlib.util
import json
import os
import os.path as osp
import shutil
import sys
import tempfile

HERE = osp.dirname(osp.abspath(__file__))
QL_ROOT = osp.dirname(HERE)
REPO = osp.dirname(osp.dirname(QL_ROOT))
QD = osp.join(REPO, 'tools', 'query_diag')
FROZEN = ('img_backbone', 'img_neck', 'view_transform', 'pts_voxel_layer',
          'pts_voxel_encoder', 'pts_middle_encoder')
TRAINABLE = ('fusion_layer', 'pts_backbone', 'pts_neck', 'bbox_head')
PRE = 'NuScenes metric/pred_instances_3d_NuScenes/'


def load_module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def diag_common():
    return load_module('diag_common', osp.join(QD, 'diag_common.py'))


def ql_md5():
    dc = diag_common()
    out = {}
    for dp, dn, fn in os.walk(QL_ROOT):
        dn[:] = [d for d in dn if d != '__pycache__']
        for n in sorted(fn):
            if n.endswith(('.py', '.sh', '.json')):
                p = osp.join(dp, n)
                out[osp.relpath(p, QL_ROOT)] = dc.file_md5(p)
    return out


def load_cfg(config, cfg_options=None):
    from mmengine.config import Config
    cfg = Config.fromfile(config)
    if cfg_options:
        cfg.merge_from_dict(cfg_options)
    return cfg


def unwrap(model):
    return model.module if hasattr(model, 'module') else model


def param_counts(model):
    """Parameter counts per top-level module and the trainable split."""
    m = unwrap(model)
    per = {}
    for name, sub in m.named_children():
        ps = list(sub.parameters())
        per[name] = {'total': sum(p.numel() for p in ps),
                     'trainable': sum(p.numel() for p in ps
                                      if p.requires_grad)}
    total = sum(v['total'] for v in per.values())
    trainable = sum(v['trainable'] for v in per.values())
    exp_frozen = sum(per[n]['total'] for n in FROZEN if n in per)
    exp_train = sum(per[n]['total'] for n in TRAINABLE if n in per)
    leak = {n: per[n]['trainable'] for n in FROZEN
            if n in per and per[n]['trainable']}
    stuck = {n: per[n]['total'] - per[n]['trainable'] for n in TRAINABLE
             if n in per and per[n]['trainable'] != per[n]['total']}
    ok = (trainable == exp_train and total - trainable == exp_frozen and
          not leak and not stuck)
    return {'per_module': per, 'total': total, 'trainable': trainable,
            'frozen': total - trainable, 'expected_trainable': exp_train,
            'expected_frozen': exp_frozen, 'frozen_with_grad': leak,
            'trainable_without_grad': stuck, 'ok': ok}


def write_report(out_dir, stem, lines, res):
    os.makedirs(out_dir, exist_ok=True)
    text = '\n'.join(lines)
    print(text)
    with open(osp.join(out_dir, stem + '.txt'), 'w') as f:
        f.write(text + '\n')
    with open(osp.join(out_dir, stem + '.json'), 'w') as f:
        json.dump(res, f, indent=2, default=str)
    print('[queryloc] saved %s/%s.{txt,json}' % (out_dir, stem))


def breakdown(ref_json, run_json):
    """oracle_breakdown.py (unchanged) of one run against the E0 json."""
    sys.path.insert(0, QD)
    import oracle_breakdown
    tmp = tempfile.mkdtemp(prefix='ql_bd_')
    try:
        for p in (ref_json, run_json):
            with open(p) as f:
                tag = json.load(f)['args']['tag']
            d = osp.join(tmp, 'oracle_' + tag)
            os.makedirs(d)
            shutil.copy(p, osp.join(d, 'oracle_%s.json' % tag))
        with open(ref_json) as f:
            base = json.load(f)['args']['tag']
        return oracle_breakdown.breakdown_lines(tmp, base)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
