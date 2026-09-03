"""CPU-only isolated checks; no MMEngine/MMCV, CUDA, data or real Runner.

Run: python -m pytest -q projects/EPFusion/tests/test_block1_archival.py
AST extraction executes the source helpers, not a reimplementation of them.
"""
import argparse
import ast
import copy
import datetime
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / 'projects/EPFusion/scripts/sanity_lambda_logging.py'


@pytest.fixture
def source():
    names = {'Tee', 'read_version', 'load_ckpt_loose', 'detect_training_processes',
             'feature_scale', 'mode_loss_summary', 'check_r0_bn_eval', 'main'}
    tree = ast.parse(SCRIPT.read_text(encoding='utf-8'))
    tree.body = [node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                 and node.name in names]
    namespace = dict(
        argparse=argparse, copy=copy, datetime=datetime, hashlib=hashlib,
        math=math, os=os, subprocess=subprocess, sys=sys, __file__=str(SCRIPT),
        REPO_ROOT=str(ROOT), DictAction=argparse.Action)
    exec(compile(tree, str(SCRIPT), 'exec'), namespace)
    return SimpleNamespace(**namespace), namespace


def test_feature_scale_readonly(source):
    src, _ = source
    ft = torch.tensor([-2., 0., 4.], requires_grad=True)
    before = ft.detach().clone()
    assert src.feature_scale(ft) == 'F_T_abs_mean=2 F_T_abs_max=4'
    assert torch.equal(ft.detach(), before) and ft.grad is None


@pytest.mark.parametrize('weight', [0.1, 1.0])
def test_teacher_ratios_separate_weighted_and_raw(source, weight):
    src, _ = source
    logs = dict(loss=2 + 3 * weight, loss_teach=3 * weight, teach_nll_raw=3,
                loss_cls=0.5, layer_0_loss_bbox=1.5, matched_ious=0.1)
    detail = src.mode_loss_summary('clean', logs, weight)
    assert 'bbox_loss_sum=2' in detail
    assert 'raw_to_bbox=1.5' in detail
    assert 'teacher_to_bbox=%.9g' % (1.5 * weight) in detail
    assert 'matched_ious' not in detail


def test_zero_bbox_has_no_invented_ratio(source):
    src, _ = source
    logs = dict(loss=1., loss_teach=1., teach_nll_raw=1., loss_bbox=0.)
    assert 'raw_to_bbox=undefined' in src.mode_loss_summary('clean', logs, 1.)


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_nonfinite_loss_rejected(source, value):
    src, _ = source
    with pytest.raises(ValueError, match='non-finite'):
        src.mode_loss_summary('clean', dict(
            loss=value, loss_teach=value, teach_nll_raw=value, loss_bbox=1.), 1.)


def make_r0(norm):
    """Use EPFusion's actual train/_eval_student_fuser_norm on a tiny CPU model."""
    path = ROOT / 'projects/EPFusion/epfusion/ep_fusion.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    cls.decorator_list = []
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef)
                and node.name in ('train', '_eval_student_fuser_norm')]
    tree.body = [cls]
    namespace = dict(BEVFusion=torch.nn.Module,
                     _BatchNorm=torch.nn.modules.batchnorm._BatchNorm)
    exec(compile(tree, str(path), 'exec'), namespace)
    model = namespace['EPFusion']()
    model.FROZEN_MODULES = ('fusion_layer',)
    model.fusion_layer = torch.nn.Sequential(norm(2))
    model.student_fuser = torch.nn.Sequential(norm(2))
    return model


@pytest.mark.parametrize('norm', [torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm])
def test_r0_eval_after_repeated_train_calls(source, norm):
    src, _ = source
    model = make_r0(norm)
    for _ in range(2):
        model.eval()
        ok, detail = src.check_r0_bn_eval(model)
        assert ok and 'bn_count=1 bn_training=[] bn_frozen_affine=[]' in detail
        assert not model.fusion_layer[0].training
        assert model.student_fuser[0].weight.requires_grad


def test_r0_training_bn_and_frozen_affine_rejected(source):
    src, _ = source
    wrong = torch.nn.Module()
    wrong.student_fuser = torch.nn.Sequential(torch.nn.BatchNorm2d(2))
    assert not src.check_r0_bn_eval(wrong)[0]
    model = make_r0(torch.nn.BatchNorm2d)
    model.student_fuser[0].weight.requires_grad_(False)
    assert not src.check_r0_bn_eval(model)[0]
    model.student_fuser = torch.nn.Identity()
    assert not src.check_r0_bn_eval(model)[0]


@pytest.mark.parametrize('failure', ['training', 'gpu', 'ps_error', 'smi_error', 'smi_missing'])
def test_process_scan_is_fail_closed(source, monkeypatch, failure):
    src, _ = source
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')

    def run(cmd, **kwargs):
        if cmd[0] == 'ps':
            return SimpleNamespace(returncode=int(failure == 'ps_error'),
                                   stdout='1 python tools/train.py' if failure == 'training' else '')
        assert cmd[-2:] == ['-i', '0']
        if failure == 'smi_missing':
            raise FileNotFoundError()
        return SimpleNamespace(returncode=int(failure == 'smi_error'),
                               stdout='2, python, 1000 MiB' if failure == 'gpu' else '')

    monkeypatch.setattr(subprocess, 'run', run)
    assert src.detect_training_processes()


def run_main(source, monkeypatch, argv):
    src, _ = source
    previous = sys.stdout
    monkeypatch.setattr(sys, 'argv', ['sanity_lambda_logging.py'] + argv)
    try:
        with pytest.raises(SystemExit) as error:
            src.main()
        return error.value.code
    finally:
        if isinstance(sys.stdout, src.Tee):
            sys.stdout.flush()
            sys.stdout.file.close()
        sys.stdout = previous


def test_hash_mismatch_stops_before_process_scan(source, monkeypatch, tmp_path):
    _, namespace = source
    namespace['detect_training_processes'] = lambda: pytest.fail('scan must not run')
    code = run_main(source, monkeypatch, [
        '--config', 'unused.py', '--out-dir', str(tmp_path / 'new'),
        '--source-sha', 'a' * 40, '--expected-script-sha256', '0' * 64])
    assert code == 2
    report = next((tmp_path / 'new').glob('sanity_*.txt')).read_text(encoding='utf-8')
    assert 'SCRIPT_SHA256_CHECK: FAIL' in report


def test_busy_gpu_stops_before_build_and_reports_do_not_overwrite(source, monkeypatch, tmp_path):
    _, namespace = source
    namespace['detect_training_processes'] = lambda: ['GPU busy']
    namespace['build_model_from_cfg'] = lambda *a, **k: pytest.fail('must not build')
    digest = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    for _ in range(2):
        assert run_main(source, monkeypatch, [
            '--config', 'unused.py', '--out-dir', str(tmp_path),
            '--expected-script-sha256', digest]) == 2
    reports = list(tmp_path.glob('sanity_*.txt'))
    assert len(reports) == 2
    for report in reports:
        content = report.read_text(encoding='utf-8')
        assert 'SCRIPT_SHA256_CHECK: PASS' in content and 'BLOCKED' in content


def test_invalid_source_sha_rejected(source, monkeypatch, tmp_path):
    assert run_main(source, monkeypatch, [
        '--config', 'unused.py', '--source-sha', 'branch-name',
        '--out-dir', str(tmp_path)]) == 2
    assert not list(tmp_path.iterdir())


def test_missing_checkpoint_never_uses_random_weights(source, tmp_path):
    src, _ = source
    with pytest.raises(FileNotFoundError):
        src.load_ckpt_loose(torch.nn.Linear(1, 1), str(tmp_path / 'missing.pth'))
