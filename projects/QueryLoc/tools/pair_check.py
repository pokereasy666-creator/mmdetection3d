#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exec doc #3 paired checks on the server (1 GPU).

sm1  S-M1: one training batch, C0 (original head) vs M-b with zero offsets,
     both from the start weights. The backbone features are computed once
     (train mode, frozen parts in eval) and fed to both heads, so the
     nondeterministic sparse-conv / BEV-pool kernels cannot differ between
     the two; heads run with the same RNG seed (dropout), cuDNN
     deterministic. Every loss except loss_offset must be identical, under
     AMP (as in training) and in fp32.
st5  S-T5 (step 2): original head vs a QueryLoc head on the SAME weights,
     --frames val frames: selected cells, query_labels and
     query_heatmap_score identical (cells of the original head are
     re-derived with diag_common.replicate_init, which S-0a2 showed to be
     exact).
Usage: python projects/QueryLoc/tools/pair_check.py sm1 --ckpt CKPT
       python projects/QueryLoc/tools/pair_check.py st5 --ckpt W \
           --config-b ql_mb [--frames 300]
"""
import argparse
import copy
import os.path as osp
import sys
import time

import torch

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import ql_common as qc  # noqa: E402


def build(name, ckpt, cfg_options):
    from mmengine.registry import init_default_scope
    from mmengine.runner.checkpoint import load_checkpoint

    from mmdet3d.registry import MODELS
    cfg = qc.load_cfg(osp.join(qc.QL_ROOT, 'configs', name + '.py'),
                      cfg_options)
    init_default_scope('mmdet3d')
    model = MODELS.build(cfg.model)
    load_checkpoint(model, ckpt, map_location='cpu', strict=False)
    return cfg, model.cuda()


def head_losses(model, feats, samples, seed, amp):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16,
                                         enabled=amp):
        return {k: v.detach().float().cpu() if torch.is_tensor(v) else v
                for k, v in model.bbox_head.loss(feats, samples).items()}


def cmd_sm1(args):
    from mmengine.runner import Runner
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cfg, c0 = build('ql_c0', args.ckpt, args.cfg_options)
    _, mb = build('ql_mb', args.ckpt, args.cfg_options)
    c0.train()
    mb.train()
    dl = Runner.build_dataloader(cfg.train_dataloader, seed=args.seed)
    batch = next(iter(dl))
    L = ['Exec doc #3 S-M1 (server): one training batch, C0 vs zero-offset '
         'M-b, same features, same RNG seed']
    rows, ok = [], True
    for amp in (True, False):
        with torch.no_grad():
            data = c0.data_preprocessor(copy.deepcopy(batch), True)
            metas = [s.metainfo for s in data['data_samples']]
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                feats = c0.extract_feat(data['inputs'], metas)
        la = head_losses(c0, feats, data['data_samples'], args.seed, amp)
        lb = head_losses(mb, feats, data['data_samples'], args.seed, amp)
        for k in sorted(la):
            eq = bool(torch.equal(torch.as_tensor(la[k]),
                                  torch.as_tensor(lb[k])))
            d = float((torch.as_tensor(la[k]) -
                       torch.as_tensor(lb[k])).abs().max())
            rows.append({'amp': amp, 'loss': k, 'c0': float(la[k]),
                         'mb': float(lb[k]), 'equal': eq, 'max_abs_diff': d})
            ok &= eq
            L.append('  amp=%-5s %-22s C0 %.8f  M-b %.8f  %s'
                     % (amp, k, float(la[k]), float(lb[k]),
                        'identical' if eq else 'DIFF %.3g' % d))
        off = lb.get('loss_offset')
        L.append('  amp=%-5s loss_offset (M-b only) %s' % (amp, off))
        ok &= off is not None and bool(torch.isfinite(
            torch.as_tensor(off)).all())
    L.append('S-M1  %s' % ('PASS' if ok else 'FAIL'))
    res = {'rows': rows, 'result': 'PASS' if ok else 'FAIL',
           'seed': args.seed, 'queryloc_md5': qc.ql_md5()}
    qc.write_report(args.out_dir, 'sm1', L, res)
    return 0 if ok else 1


def cmd_st5(args):
    from mmengine.runner import Runner
    dc = qc.diag_common()
    cfg, ma = build('ql_c0', args.ckpt, args.cfg_options)
    _, mb = build(args.config_b, args.ckpt, args.cfg_options)
    ma.eval()
    mb.eval()
    cap = {}

    def logit_hook(module, inputs, output):
        cap['logits_a'] = output.detach().clone()

    ma.bbox_head.heatmap_head.register_forward_hook(logit_hook)
    dl = Runner.build_dataloader(cfg.test_dataloader)
    n = same_cells = same_labels = same_scores = 0
    P = int(ma.bbox_head.num_proposals)
    with torch.no_grad():
        for data in dl:
            if n >= args.frames:
                break
            d = ma.data_preprocessor(copy.deepcopy(data), False)
            metas = [s.metainfo for s in d['data_samples']]
            # one feature computation for both heads (GPU sparse-conv /
            # BEV-pool kernels are not deterministic across two passes)
            feats = ma.extract_feat(d['inputs'], metas)
            out_a = dc.first_dict(ma.bbox_head(feats, metas))
            out_b = dc.first_dict(mb.bbox_head(feats, metas))
            idx, cls, _ = dc.replicate_init(ma.bbox_head, cap['logits_a'], P)
            b = len(d['data_samples'])
            n += b
            same_cells += b * int(torch.equal(
                idx, mb.bbox_head.query_cells_init))
            same_labels += b * int(torch.equal(ma.bbox_head.query_labels,
                                               mb.bbox_head.query_labels))
            same_scores += b * int(torch.equal(
                out_a['query_heatmap_score'], out_b['query_heatmap_score']))
    ok = n > 0 and same_cells == same_labels == same_scores == n
    L = ['Exec doc #3 S-T5: original head vs %s on %s, %d frames (same '
         'features)' % (args.config_b, args.ckpt, n),
         '  identical selected cells %d / %d, labels %d / %d, '
         'query_heatmap_score %d / %d' % (same_cells, n, same_labels, n,
                                          same_scores, n),
         'S-T5  %s' % ('PASS' if ok else 'FAIL')]
    res = {'frames': n, 'same_cells': same_cells, 'same_labels': same_labels,
           'same_scores': same_scores, 'result': 'PASS' if ok else 'FAIL',
           'config_b': args.config_b, 'ckpt': args.ckpt,
           'queryloc_md5': qc.ql_md5()}
    qc.write_report(args.out_dir, 'st5_' + args.config_b, L, res)
    return 0 if ok else 1


def main():
    from mmengine.config import DictAction
    p = argparse.ArgumentParser(description='exec doc #3 paired checks')
    p.add_argument('mode', choices=['sm1', 'st5'])
    p.add_argument('--ckpt', required=True)
    p.add_argument('--config-b', default='ql_mb')
    p.add_argument('--frames', type=int, default=300)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out-dir', default='work_dirs/queryloc/step0/pair')
    p.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = p.parse_args()
    t0 = time.time()
    rc = cmd_sm1(args) if args.mode == 'sm1' else cmd_st5(args)
    print('[queryloc] %s done in %.1fs' % (args.mode, time.time() - t0))
    return rc


if __name__ == '__main__':
    sys.exit(main())
