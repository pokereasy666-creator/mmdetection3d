#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exec doc #3 sec. 4.2 rehearsal + S-T2 (not an experiment).

Runs the real training loop of one config (runner, AMP, accumulation,
SyncBN, DDP when launched with torchrun) for --iters dataloader iterations
from the start weights, then stops (no checkpoint, no validation):
  * s/iter over iterations (skip, iters], peak GPU memory per rank,
    iterations per epoch and the extrapolated wall time of one epoch;
  * trainable / frozen parameter counts (sec. 3 split);
  * effective batch = world size x batch per GPU x accumulative counts,
    the learning rate and Adam betas actually in the optimizer;
  * S-T2: frozen modules (parameters AND buffers) identical before and
    after, frozen modules in eval mode during training, trainable
    parameters did change, no NaN / Inf in the logged losses; the start
    state of new modules as S-T0 on the real training path (after
    init_weights and load_from): M-b last offset conv zero, D3 layers 1-2
    equal to layer 0.
Usage (run_queryloc.sh rehearse <name> wraps it):
  torchrun --nproc_per_node=4 projects/QueryLoc/tools/rehearse.py \
      projects/QueryLoc/configs/ql_c0.py --ckpt CKPT --launcher pytorch \
      --work-dir work_dirs/queryloc/step0/rehearse/ql_c0 [--cfg-options ...]
"""
import argparse
import math
import os.path as osp
import sys
import time

import torch

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import ql_common as qc  # noqa: E402


class RehearsalDone(Exception):
    pass


def start_state(head):
    """S-T0 init checks on the real training path (init_weights, then
    load_from), read in before_train."""
    out = {}
    if getattr(head, 'offset_head', None) is not None:
        last = head.offset_head[-1]
        out['offset_head last conv all zero'] = bool(
            not last.weight.any() and not last.bias.any())
    if int(head.num_decoder_layers) > 1:
        for sub in ('decoder', 'prediction_heads'):
            s0 = getattr(head, sub)[0].state_dict()
            for i in range(1, int(head.num_decoder_layers)):
                si = getattr(head, sub)[i].state_dict()
                out['%s.%d == %s.0' % (sub, i, sub)] = set(s0) == set(si) \
                    and all(torch.equal(s0[k], si[k]) for k in s0)
    return out


def make_hook(args, stats):
    from mmengine.hooks import Hook

    class RehearsalHook(Hook):
        priority = 'VERY_LOW'

        def before_train(self, runner):
            m = qc.unwrap(runner.model)
            stats['frozen_before'] = {
                k: v.detach().cpu().clone()
                for name in qc.FROZEN if hasattr(m, name)
                for k, v in getattr(m, name).state_dict(
                    prefix=name + '.').items()}
            stats['train_before'] = {
                k: p.detach().cpu().clone()
                for k, p in m.named_parameters() if p.requires_grad}
            stats['params'] = qc.param_counts(m)
            stats['start_state'] = start_state(m.bbox_head)
            opt = runner.optim_wrapper.optimizer
            stats['optimizer'] = [{'lr': g['lr'], 'betas': g.get('betas'),
                                   'weight_decay': g.get('weight_decay'),
                                   'n_params': len(g['params'])}
                                  for g in opt.param_groups]
            stats['accumulative_counts'] = getattr(
                runner.optim_wrapper, '_accumulative_counts', None)
            stats['iters_per_epoch'] = len(runner.train_dataloader)
            stats['batch_per_gpu'] = runner.train_dataloader.batch_size
            stats['losses'] = []

        def before_train_iter(self, runner, batch_idx, data_batch=None):
            it = runner.iter
            if it == 0:
                m = qc.unwrap(runner.model)
                stats['frozen_training'] = {
                    n: bool(getattr(m, n).training) for n in qc.FROZEN
                    if hasattr(m, n)}
                stats['trainable_training'] = {
                    n: bool(getattr(m, n).training) for n in qc.TRAINABLE
                    if hasattr(m, n)}
            if it == args.skip:
                torch.cuda.synchronize()
                stats['t_start'] = time.time()

        def after_train_iter(self, runner, batch_idx, data_batch=None,
                             outputs=None):
            vals = {k: float(v) for k, v in (outputs or {}).items()
                    if torch.is_tensor(v) or isinstance(v, float)}
            stats['losses'].append(vals)
            if runner.iter + 1 >= args.iters:
                torch.cuda.synchronize()
                stats['t_end'] = time.time()
                stats['lr_end'] = [g['lr'] for g in
                                   runner.optim_wrapper.optimizer.param_groups]
                stats['peak_mem_mib'] = round(
                    torch.cuda.max_memory_allocated() / 2.0**20, 1)
                raise RehearsalDone()

    return RehearsalHook()


def main():
    from mmengine.config import DictAction
    p = argparse.ArgumentParser(description='exec doc #3 rehearsal')
    p.add_argument('config')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--work-dir', required=True)
    p.add_argument('--iters', type=int, default=100)
    p.add_argument('--skip', type=int, default=20,
                   help='iterations excluded from the timing')
    p.add_argument('--launcher', default='none')
    p.add_argument('--local_rank', '--local-rank', type=int, default=0)
    p.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = p.parse_args()
    from mmengine.dist import all_gather_object, get_rank, get_world_size
    from mmengine.runner import Runner

    t0 = time.time()
    cfg = qc.load_cfg(args.config, args.cfg_options)
    cfg.work_dir = args.work_dir
    cfg.launcher = args.launcher
    cfg.load_from = args.ckpt
    cfg.resume = False
    runner = Runner.from_cfg(cfg)
    stats = {}
    runner.register_hook(make_hook(args, stats))
    try:
        runner.train()
        stats['error'] = 'training ended before --iters'
    except RehearsalDone:
        pass
    m = qc.unwrap(runner.model)
    after = {k: v.detach().cpu() for name in qc.FROZEN if hasattr(m, name)
             for k, v in getattr(m, name).state_dict(
                 prefix=name + '.').items()}
    changed = sorted(k for k, v in stats['frozen_before'].items()
                     if not torch.equal(v, after[k]))
    moved = sum(1 for k, p in m.named_parameters()
                if p.requires_grad and
                not torch.equal(p.detach().cpu(), stats['train_before'][k]))
    peaks = all_gather_object(stats.get('peak_mem_mib'))
    if get_rank() != 0:
        return 0
    world = get_world_size()
    n_timed = args.iters - args.skip
    s_iter = (stats['t_end'] - stats['t_start']) / max(n_timed, 1)
    epoch_h = s_iter * stats['iters_per_epoch'] / 3600.0
    bad_loss = [i for i, d in enumerate(stats['losses'])
                if any(not math.isfinite(v) for v in d.values())]
    eff = world * stats['batch_per_gpu'] * (stats['accumulative_counts'] or 1)
    st2 = {'frozen_unchanged': not changed,
           'frozen_eval_during_training':
           not any(stats.get('frozen_training', {1: True}).values()),
           'trainable_in_train_mode':
           all(stats.get('trainable_training', {1: False}).values()),
           'trainable_params_changed': moved,
           'param_split_as_sec3': stats['params']['ok'],
           'start_state_as_st0': all(stats['start_state'].values()),
           'losses_finite': not bad_loss}
    ok = (st2['frozen_unchanged'] and st2['frozen_eval_during_training'] and
          st2['trainable_in_train_mode'] and moved > 0 and
          st2['param_split_as_sec3'] and st2['start_state_as_st0'] and
          not bad_loss)
    name = osp.splitext(osp.basename(args.config))[0]
    c = stats['params']
    L = ['Exec doc #3 sec. 4.2 rehearsal: %s (%d iters, timed %d-%d)'
         % (name, args.iters, args.skip, args.iters),
         '  world size %d  batch/GPU %d  accumulative %s  effective batch %d'
         % (world, stats['batch_per_gpu'], stats['accumulative_counts'],
            eff),
         '  s/iter %.4f  iters/epoch %d  -> %.2f h per epoch'
         % (s_iter, stats['iters_per_epoch'], epoch_h),
         '  peak memory per rank (MiB): %s' % peaks,
         '  params: total %d  trainable %d  frozen %d'
         % (c['total'], c['trainable'], c['frozen']),
         '  optimizer groups: %d; first lr %.3g -> %.3g at iter %d; betas %s'
         % (len(stats['optimizer']), stats['optimizer'][0]['lr'],
            stats['lr_end'][0], args.iters, stats['optimizer'][0]['betas']),
         '  start state (after init_weights + load_from): %s'
         % (stats['start_state'] or 'no new modules'),
         '  S-T2: %s' % ', '.join('%s=%s' % kv for kv in st2.items()),
         'S-T2  %s' % ('PASS' if ok else 'FAIL')]
    if changed:
        L.insert(-1, '  frozen tensors changed: %s' % changed[:5])
    res = {'config': name, 'iters': args.iters, 'skip': args.skip,
           'world_size': world, 'effective_batch': eff, 's_per_iter': s_iter,
           'iters_per_epoch': stats['iters_per_epoch'],
           'epoch_hours': epoch_h, 'peak_mem_mib_per_rank': peaks,
           'params': c, 'optimizer': stats['optimizer'],
           'lr_end': stats['lr_end'], 'st2': st2, 'st2_changed': changed,
           'start_state': stats['start_state'],
           'bad_loss_iters': bad_loss, 'losses_first_last':
           [stats['losses'][0], stats['losses'][-1]],
           'result': 'PASS' if ok else 'FAIL',
           'elapsed_s': round(time.time() - t0, 1),
           'queryloc_md5': qc.ql_md5(),
           'repo_version': qc.diag_common().read_version_line()}
    qc.write_report(args.work_dir, 'rehearse_' + name, L, res)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
