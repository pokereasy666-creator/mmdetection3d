#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""sanity_lambda_logging.py —— 块1 验收闸门（M0_PLAN G-1）。

只读冒烟 + 断言，不训练、不落盘（除单文件报告）。需 GPU + nuScenes（服务器执行）。
本脚本不经 Runner.train()，故 R0 权重相等断言前【显式调用】hooks.copy_weights(model)。

断言项（a-i）：
  a) 12 个 Λ 键 + loss_teach/teach_nll_raw + 4 个坍缩预警键全部在场；
  b) 初始化下 clean 模式 mean(Λ_C)≈mean(Λ_L)≈1（poe_fuser 末层零初始化，容差 1e-6）；
  c) train pipeline cfg 无 'ObjectSample'（铁律 4）；
  d) 教师路径受控 allclose：EP 的 clean F_T vs 原 BEVFusion fusion_layer 输出
     （同 ckpt / 双 eval / fp32 / 同一固定 batch，rtol1e-4/atol1e-5）；
  e) R0：load_checkpoint 后显式 copy_weights(model)，断言 student_fuser≡fusion_layer 逐张量相等；
  f) 单卡 backward 冒烟：一次反传后可训练参数 grad 非 None、冻结参数 grad 为 None；
  g) 'zero_image'/'zero_points' 强制路径冒烟（不崩溃、Λ 键在场）；
  h) 打印 scheduler 构造后的 LR/momentum，并断言 epoch 端点不越界；
  i) Λ≡1 时 PoE 教师初始化输出与冻结 ConvFuser 输出 allclose。
     默认使用原后端；显式 --equivalence-backend no_cudnn 仅切换 i 的融合层验收后端，
     保留原后端比较记录，容差不变，非有限值及运行异常仍为硬失败。

用法（服务器，复现训练结束后）：
  bash scripts/m0_probe.sh  # 确认环境（可选）
  python projects/EPFusion/scripts/sanity_lambda_logging.py \
      --config projects/EPFusion/configs/epfusion_m0_poe_4xa30-amp-accum_nus-3d.py \
      --r0-config projects/EPFusion/configs/epfusion_m0_r0_convfuser_4xa30-amp-accum_nus-3d.py \
      --checkpoint work_dirs/bevfusion_lidar-cam_official6e_4xa30_amp512_accum4_seed577127641/epoch_5.pth \
      --device cuda:0
"""
import argparse
import copy
import datetime
import hashlib
import os
import subprocess
import sys
import traceback
from contextlib import contextmanager

# PyTorch deterministic CUDA GEMM requires this to be set before CUDA starts.
# Respect an explicit caller choice (the other supported value is ``:16:8``).
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mmengine.config import Config, DictAction


class Tee(object):
    def __init__(self, path):
        self.file = open(path, 'w', encoding='utf-8')
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def read_version():
    try:
        with open(os.path.join(REPO_ROOT, 'VERSION'), encoding='utf-8') as f:
            return f.read().strip().splitlines()[0]
    except Exception:
        return 'unknown'


def apply_data_root(cfg, data_root):
    """递归覆盖 dataloader/evaluator 中已有的 data_root 字段。

    已在 config parse 时拼接好的 ann_file 字符串不会随之改变；切换非默认
    数据根时，须同时用 --cfg-options 覆盖对应 ann_file。
    """
    if data_root is None:
        return
    cfg.data_root = data_root

    def _patch(value):
        if isinstance(value, dict):
            if 'data_root' in value:
                value['data_root'] = data_root
            for child in value.values():
                _patch(child)
        elif isinstance(value, list):
            for child in value:
                _patch(child)

    for key in ('train_dataloader', 'val_dataloader', 'test_dataloader',
                'val_evaluator', 'test_evaluator'):
        if key in cfg:
            _patch(cfg[key])


def build_model_from_cfg(config_path, cfg_options=None, data_root=None):
    """构建模型（离线兼容：置空 img_backbone.init_cfg，阻止 Swin 联网下载）。"""
    from mmengine.registry import init_default_scope
    cfg = Config.fromfile(config_path)
    if cfg_options:
        cfg.merge_from_dict(cfg_options)
    apply_data_root(cfg, data_root)
    ci = cfg.get('custom_imports', None)
    if ci:
        from mmengine.utils import import_modules_from_strings
        import_modules_from_strings(**ci)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    from mmdet3d.registry import MODELS
    model_cfg = copy.deepcopy(cfg.model)
    img_bb = model_cfg.get('img_backbone', None)
    if isinstance(img_bb, dict) and img_bb.get('init_cfg', None) is not None:
        img_bb['init_cfg'] = None
    model = MODELS.build(model_cfg)
    return cfg, model


def load_ckpt_loose(model, path):
    import torch
    if not path or not os.path.isfile(path):
        print('  [warn] ckpt 不存在，使用随机/零初始化：%s' % path)
        return
    sd = torch.load(path, map_location='cpu')
    sd = sd.get('state_dict', sd)
    sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
    res = model.load_state_dict(sd, strict=False)
    print('  loaded ckpt: missing=%d unexpected=%d'
          % (len(res.missing_keys), len(res.unexpected_keys)))
    print('  missing_keys=%s unexpected_keys=%s'
          % (res.missing_keys, res.unexpected_keys))


def build_train_batch(cfg, num=1):
    from mmengine.runner import Runner
    dl_cfg = copy.deepcopy(cfg.train_dataloader)
    ds = dl_cfg['dataset']
    if ds.get('type') == 'CBGSDataset':
        ds = ds['dataset']
    dl_cfg['dataset'] = ds
    dl_cfg['batch_size'] = 2
    dl_cfg['num_workers'] = 2
    dl_cfg['persistent_workers'] = False
    loader = Runner.build_dataloader(dl_cfg)
    return next(iter(loader)), len(loader)


def detect_training_processes():
    """检测训练/GPU compute 进程；逻辑复用 scripts/m0_probe.py:110-152。"""
    suspects = []
    try:
        out = subprocess.run(
            ['ps', '-eo', 'pid,args'], capture_output=True, text=True,
            timeout=15).stdout
        for line in out.splitlines():
            low = line.lower()
            if 'sanity_lambda_logging.py' in low:
                continue
            if any(k in low for k in (
                    'tools/train.py', 'torchrun', 'torch.distributed.launch',
                    'torch.distributed.run', 'dist_train')):
                suspects.append('[ps] ' + line.strip()[:200])
    except Exception as e:
        suspects.append('[ps] 进程扫描失败（无法确认）: %r' % e)

    cvd = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if cvd is not None and cvd.strip() == '':
        return suspects
    try:
        cmd = [
            'nvidia-smi',
            '--query-compute-apps=pid,process_name,used_memory',
            '--format=csv,noheader',
        ]
        if cvd is not None and cvd.strip():
            cmd += ['-i', cvd.strip()]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            scope = ('可见 GPU [%s]' % cvd.strip()
                     if (cvd and cvd.strip()) else '全部 GPU')
            for line in result.stdout.strip().splitlines():
                if line.strip():
                    suspects.append(
                        '[gpu] compute 进程(%s): %s'
                        % (scope, line.strip()))
    except FileNotFoundError:
        pass
    except Exception as e:
        suspects.append('[gpu] nvidia-smi 查询失败（无法确认）: %r' % e)
    return suspects


LAMBDA_KEYS = ['lambda_%s_%s_%s' % (br, m, k)
               for br in ('C', 'L')
               for m in ('clean', 'corrupt_cam', 'corrupt_lidar')
               for k in ('sum', 'cnt')]
WARN_KEYS = ['proj_var_C', 'proj_var_L', 'cos_pc_ft', 'cos_pl_ft']


@contextmanager
def strict_fp32(deterministic=False):
    """等价性检查禁用 TF32/AMP，并可临时启用确定性算法。"""
    import torch
    mm, cudnn = torch.backends.cuda.matmul, torch.backends.cudnn
    old = (mm.allow_tf32, cudnn.allow_tf32, cudnn.benchmark,
           cudnn.deterministic)
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    get_warn_only = getattr(
        torch, 'is_deterministic_algorithms_warn_only_enabled', None)
    old_warn_only = get_warn_only() if get_warn_only is not None else False
    try:
        mm.allow_tf32 = cudnn.allow_tf32 = False
        cudnn.benchmark = False
        cudnn.deterministic = True
        if deterministic:
            # warn_only=False：无确定性实现时验收必须显式 ERROR，不能静默通过。
            torch.use_deterministic_algorithms(True, warn_only=False)
        with torch.autocast(device_type='cuda', enabled=False):
            yield
    finally:
        if deterministic:
            torch.use_deterministic_algorithms(
                old_deterministic, warn_only=old_warn_only)
        (mm.allow_tf32, cudnn.allow_tf32, cudnn.benchmark,
         cudnn.deterministic) = old


def compare_tensor(name, actual, reference, exact=False):
    """原 allclose 阈值不变；统计在 CPU 上完成，避免诊断再占 GPU。"""
    import torch
    a, b = actual.detach().cpu(), reference.detach().cpu()
    if a.shape != b.shape or a.dtype != b.dtype:
        return False, '%s shape/dtype mismatch: %s/%s vs %s/%s' % (
            name, tuple(a.shape), a.dtype, tuple(b.shape), b.dtype)
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    close = (a == b) if exact else torch.isclose(a, b, rtol=1e-4, atol=1e-5)
    ok = finite and bool(close.all())
    diff = (a.double() - b.double()).abs()
    max_diff = float(diff.max()) if diff.numel() else 0.0
    bad = int((~close).sum())
    return ok, '%s ok=%s finite=%s max_abs_diff=%.3e bad=%d/%d (%s)' % (
        name, ok, finite, max_diff, bad, a.numel(),
        'exact' if exact else 'rtol1e-4/atol1e-5')


def teacher_fuser_forward(fuser, F_C, F_L):
    """运行原 fuser；只读记录 Conv/BN 输出，BN 必须在原地 ReLU 前复制。"""
    stages, handles = {}, []

    def _capture(name):
        def _hook(module, inputs, output):
            stages[name] = output.detach().cpu().clone()
        return _hook

    try:
        for name, module in (('conv', fuser[0]), ('bn', fuser[1])):
            handles.append(module.register_forward_hook(_capture(name)))
        output = fuser([F_C, F_L]).float()
    finally:
        for handle in handles:
            handle.remove()
    return output, stages


def poe_fp64_diagnostic(model, F_C, F_L, mu, teacher, limit=2048,
                       stages=None, bad_coords=None):
    """在 FP32 超差坐标上分解折叠权重与卷积累加误差。

    只读取实际特征和权重，不替换验收输出，也不改变原 allclose 闸门。
    """
    import torch

    if bad_coords is None:
        close = torch.isclose(mu.detach(), teacher.detach(),
                              rtol=1e-4, atol=1e-5)
        bad_coords = (~close).nonzero(as_tuple=False).detach().cpu()
    total_bad = int(bad_coords.shape[0])
    if total_bad == 0:
        return 'fp64_diag skipped (no failing coordinates)'
    bad_coords = bad_coords[:limit]

    fusion = model.fusion_layer
    conv, bn = fusion[0], fusion[1]
    poe = model.poe_fuser
    if (tuple(conv.kernel_size) != (3, 3)
            or tuple(conv.padding) != (1, 1)
            or tuple(conv.stride) != (1, 1)
            or tuple(conv.dilation) != (1, 1)
            or conv.groups != 1):
        return 'fp64_diag unavailable: unsupported teacher convolution geometry'

    # 大特征只搬 CPU fp32；每个超差位置的 3x3 patch 再转 double。
    fc = F_C.detach().cpu()
    fl = F_L.detach().cpu()
    teacher_out = teacher.detach().cpu()
    mu_out = mu.detach().cpu()
    teacher_w = conv.weight.detach().cpu()
    teacher_bias = (None if conv.bias is None
                    else conv.bias.detach().cpu().double())
    proj_c_w = poe.proj_C.weight.detach().cpu()
    proj_l_w = poe.proj_L.weight.detach().cpu()
    proj_c_b = poe.proj_C.bias.detach().cpu().double()
    proj_l_b = poe.proj_L.bias.detach().cpu().double()
    scale = (bn.weight.detach().cpu().double()
             / torch.sqrt(bn.running_var.detach().cpu().double() + bn.eps))
    folded_bias = (bn.bias.detach().cpu().double()
                   - bn.running_mean.detach().cpu().double() * scale)
    eps = float(poe.eps)

    def _dot_at(feature, weight, b, out_ch, y, x):
        height, width = feature.shape[-2:]
        fy0, fy1 = max(0, y - 1), min(height, y + 2)
        fx0, fx1 = max(0, x - 1), min(width, x + 2)
        wy0, wx0 = fy0 - (y - 1), fx0 - (x - 1)
        wy1, wx1 = wy0 + (fy1 - fy0), wx0 + (fx1 - fx0)
        patch = feature[b, :, fy0:fy1, fx0:fx1].double()
        kernel = weight[out_ch, :, wy0:wy1, wx0:wx1].double()
        return (patch * kernel).sum()

    errors = dict(
        teacher_fp32_vs_ref64=[],
        poe_fp32_vs_stored64=[],
        stored64_vs_truefold64=[],
        truefold64_vs_expected64=[],
        expected64_vs_teacher64=[])
    if stages is not None:
        errors.update(
            conv_fp32_vs_ref64=[], conv_error_after_bn_scale=[],
            bn_fp32_vs_same_input64=[], bn_fp32_vs_ref64=[])
    split = fc.shape[1]
    for coord in bad_coords.tolist():
        b, out_ch, y, x = coord
        raw_c = _dot_at(fc, teacher_w[:, :split], b, out_ch, y, x)
        raw_l = _dot_at(fl, teacher_w[:, split:], b, out_ch, y, x)
        raw = raw_c + raw_l
        if teacher_bias is not None:
            raw = raw + teacher_bias[out_ch]
        bn64 = raw * scale[out_ch] + folded_bias[out_ch]
        teacher64 = torch.relu(bn64)
        if stages is not None:
            conv32 = stages['conv'][b, out_ch, y, x].double()
            bn32 = stages['bn'][b, out_ch, y, x].double()
            bn_same_input64 = conv32 * scale[out_ch] + folded_bias[out_ch]
            errors['conv_fp32_vs_ref64'].append(float((conv32 - raw).abs()))
            errors['conv_error_after_bn_scale'].append(
                float(((conv32 - raw) * scale[out_ch]).abs()))
            errors['bn_fp32_vs_same_input64'].append(
                float((bn32 - bn_same_input64).abs()))
            errors['bn_fp32_vs_ref64'].append(float((bn32 - bn64).abs()))

        stored_c = (_dot_at(fc, proj_c_w, b, out_ch, y, x)
                    + proj_c_b[out_ch])
        stored_l = (_dot_at(fl, proj_l_w, b, out_ch, y, x)
                    + proj_l_b[out_ch])
        stored64 = torch.relu((stored_c + stored_l) / (2.0 + eps))

        true_c = 2.0 * scale[out_ch] * raw_c + folded_bias[out_ch]
        true_l = 2.0 * scale[out_ch] * raw_l + folded_bias[out_ch]
        if teacher_bias is not None:
            # Conv bias belongs once inside the unfused teacher convolution.
            true_c = true_c + 2.0 * scale[out_ch] * teacher_bias[out_ch]
        truefold64 = torch.relu((true_c + true_l) / (2.0 + eps))
        expected64 = teacher64 * (2.0 / (2.0 + eps))

        errors['teacher_fp32_vs_ref64'].append(
            abs(float(teacher_out[b, out_ch, y, x]) - float(teacher64)))
        errors['poe_fp32_vs_stored64'].append(
            abs(float(mu_out[b, out_ch, y, x]) - float(stored64)))
        errors['stored64_vs_truefold64'].append(
            abs(float(stored64) - float(truefold64)))
        errors['truefold64_vs_expected64'].append(
            abs(float(truefold64) - float(expected64)))
        errors['expected64_vs_teacher64'].append(
            abs(float(expected64) - float(teacher64)))

    return 'fp64_diag analyzed=%d/%d; %s' % (
        len(bad_coords), total_bad,
        '; '.join('%s=%.3e' % (name, max(values))
                  for name, values in errors.items()))


def fuser_backend_diagnostic(model, F_C, F_L, mu, teacher, bad_coords,
                             diagnostic_only=True):
    """仅复算两个 fuser，返回受控结果；是否用作闸门由显式 CLI 选项决定。"""
    import torch

    with strict_fp32(), torch.no_grad():
        with torch.backends.cudnn.flags(
                enabled=False, benchmark=False, deterministic=True,
                allow_tf32=False):
            native_mu = model.poe_fuser(F_C, F_L)['mu_F'].float()
            native_teacher, stages = teacher_fuser_forward(
                model.fusion_layer, F_C, F_L)
        native_finite = all(bool(torch.isfinite(t).all()) for t in (
            native_mu, native_teacher, stages['conv'], stages['bn']))
        native_ok, native_detail = compare_tensor(
            'no_cudnn_poe_vs_teacher', native_mu, native_teacher)
        msgs = [
            'backend_check diagnostic_only=%s cudnn_enabled=False finite=%s'
            % (diagnostic_only, native_finite), native_detail]
        for name, actual, reference in (
                ('teacher_no_cudnn_vs_original', native_teacher, teacher),
                ('poe_no_cudnn_vs_original', native_mu, mu)):
            _, detail = compare_tensor(name, actual, reference)
            msgs.append(detail)
        msgs.append('no_cudnn_on_original_fail_coords: ' + poe_fp64_diagnostic(
            model, F_C, F_L, native_mu, native_teacher,
            stages=stages, bad_coords=bad_coords))
    return native_ok and native_finite, '\n    '.join(msgs)


def compare_teacher_state(ep, stock):
    """逐张量检查冻结教师参数与 buffers；不比较 EP 新增模块。"""
    import torch
    mismatches = []
    for name in ep.FROZEN_MODULES:
        left = getattr(ep, name).state_dict()
        right = getattr(stock, name).state_dict()
        for key in sorted(set(left) | set(right)):
            if (key not in left or key not in right
                    or left[key].dtype != right[key].dtype
                    or not torch.equal(left[key].detach().cpu(),
                                       right[key].detach().cpu())):
                mismatches.append(name + '.' + key)
    return not mismatches, 'teacher_state mismatches=%s' % (
        mismatches or 'none')


def teacher_snapshot(model, batch, ep_path=False):
    """各自完整提取教师特征；hooks 仅记录 CPU 副本，不替换任何输入。"""
    import torch
    cap, handles = {}, []
    pp = model.data_preprocessor
    old_mode = getattr(pp, 'force_mode', None)

    def save(name, tensor):
        cap[name] = tensor.detach().cpu().clone()

    def view_input(mod, inputs):
        save('img_feats_2d', inputs[0])
        for name, tensor in zip(
                ('lidar2image', 'camera_intrinsics', 'camera2lidar',
                 'img_aug_matrix', 'lidar_aug_matrix'), inputs[2:7]):
            save('geometry.' + name, tensor)

    def output_hook(name):
        def hook(mod, inputs, output):
            save(name, output)
        return hook

    try:
        handles.append(model.view_transform.register_forward_pre_hook(
            view_input))
        for name, module in (
                ('camera_bev', model.view_transform),
                ('lidar_bev', model.pts_middle_encoder),
                ('fused', model.fusion_layer)):
            handles.append(module.register_forward_hook(output_hook(name)))
        # DepthLSSTransform 的实际深度图输入，用于定位重复像素写入差异。
        if hasattr(model.view_transform, 'dtransform'):
            handles.append(
                model.view_transform.dtransform.register_forward_pre_hook(
                    lambda mod, inputs: save('depth', inputs[0])))
        if ep_path:
            pp.force_mode = 'clean'
        with torch.no_grad():
            data = pp(copy.deepcopy(batch), False)
            inp = data['inputs']
            metas = [s.metainfo for s in data['data_samples']]
            save('inputs.imgs', inp['imgs'])
            for idx, points in enumerate(inp['points']):
                save('inputs.points.%d' % idx, points)
            if ep_path:
                geo = model._img_geo(metas, inp['imgs'])
                f2d = model._img_feats_2d(inp['imgs'])
                cam = model._img_bev(f2d, inp['points'], geo)
                lidar = model._pts_bev(inp['points'])
                model.fusion_layer([cam, lidar])
            else:
                model.extract_feat(inp, metas)
    finally:
        for handle in handles:
            handle.remove()
        if ep_path:
            pp.force_mode = old_mode
    return cap


def compare_snapshots(name, actual, reference):
    """中间特征只定位差异；输入精确一致及最终融合输出仍为验收条件。"""
    msgs, failed_stages = [], []
    all_ok = True
    stages = ['img_feats_2d']
    if 'depth' in actual or 'depth' in reference:
        stages.append('depth')
    stages += ['camera_bev', 'lidar_bev', 'fused']
    keys = sorted((set(actual) | set(reference) | {'inputs.imgs'}) - set(stages))
    keys += stages
    for key in keys:
        exact = key.startswith(('inputs.', 'geometry.'))
        if key not in actual or key not in reference:
            ok, detail = False, '%s missing snapshot key' % key
            all_ok = False
        else:
            ok, detail = compare_tensor(key, actual[key], reference[key], exact)
        if exact or key == 'fused':
            all_ok = all_ok and ok
        if not ok:
            failed_stages.append(key)
        msgs.append(detail)
    msgs.append('first_difference=%s' % (
        failed_stages[0] if failed_stages else 'none'))
    return all_ok, name + ': ' + '; '.join(msgs)


def run_mode(model, batch, mode):
    """以 force_mode 跑一次 loss()+parse_losses（不反传）。"""
    model.data_preprocessor.force_mode = mode
    try:
        data = model.data_preprocessor(copy.deepcopy(batch), True)
        losses = model.loss(data['inputs'], data['data_samples'])
        loss_sum, log_vars = model.parse_losses(losses)
        return losses, loss_sum, log_vars
    finally:
        model.data_preprocessor.force_mode = None


def check(report, name, fn):
    try:
        ok, detail = fn()
        report.append((name, 'PASS' if ok else 'FAIL', detail))
    except Exception as e:
        report.append((name, 'ERROR', repr(e) + '\n' + traceback.format_exc()))


def main():
    p = argparse.ArgumentParser(description='EP-Fusion 块1 sanity（G-1）')
    p.add_argument('--config', required=True, help='EP 主 config（poe）')
    p.add_argument('--r0-config', default=None, help='R0 对照 config（测 e 项）')
    p.add_argument('--checkpoint', default=None, help='teacher/backbone ckpt')
    p.add_argument('--base-config', default=None,
                   help='原 BEVFusion config（测 d 项；默认用 EP _base_ 推断）')
    p.add_argument('--data-root', default=None)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out-dir', default='.')
    p.add_argument(
        '--equivalence-backend', choices=('original', 'no_cudnn'),
        default='original',
        help='Backend for the i equivalence gate only; no_cudnn retains the '
             'original comparison and uses the same rtol/atol.')
    p.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default=None,
        help='Override config options, e.g. model.w_teach=0.01 '
             'randomness.seed=1')
    args = p.parse_args()

    out = os.path.join(
        args.out_dir,
        'sanity_lambda_logging_%s.txt'
        % datetime.datetime.now().strftime('%Y%m%d'))
    sys.stdout = Tee(out)
    print('EP-Fusion 块1 sanity_lambda_logging')
    print('生成时间 : %s' % datetime.datetime.now().isoformat())
    print('VERSION  : %s' % read_version())
    print('SANITY   : block1-explicit-equivalence-backend-20260903')
    with open(__file__, 'rb') as source:
        print('SCRIPT_SHA256: %s' % hashlib.sha256(source.read()).hexdigest())
    print('命令参数 : %s' % vars(args))
    process_suspects = detect_training_processes()
    if process_suspects:
        print('进程自检 : WARNING，检测到训练进程或无法确认：')
        for suspect in process_suspects:
            print('  ' + suspect)
    else:
        print('进程自检 : PASS，未发现训练/GPU compute 进程')

    import torch
    print('torch=%s TF32 before checks: matmul=%s cudnn=%s'
          % (torch.__version__, torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cudnn.allow_tf32))
    report = []

    # ---- 构建 EP（poe）模型 + 载 ckpt ----
    cfg, model = build_model_from_cfg(
        args.config,
        cfg_options=args.cfg_options,
        data_root=args.data_root)
    load_ckpt_loose(model, args.checkpoint)
    from projects.EPFusion.epfusion.hooks import init_poe_from_teacher
    ok_init = init_poe_from_teacher(model)
    model = model.to(args.device)
    model.train()  # 触发 override train()：冻结子模块强制 eval

    batch, epoch_length = build_train_batch(cfg)

    # (i) Λ≡1 时 PoE 教师初始化与冻结 ConvFuser 输出等价
    def _i():
        selected_backend = args.equivalence_backend
        if not ok_init:
            return False, 'teacher initialization not performed'
        was_training = model.training
        ep_eval = model.eval()
        ep_pp = ep_eval.data_preprocessor
        old_mode = ep_pp.force_mode
        try:
            with strict_fp32(), torch.no_grad():
                ep_pp.force_mode = 'clean'
                data = ep_pp(copy.deepcopy(batch), False)
                ep_pp.force_mode = None
                metas = [s.metainfo for s in data['data_samples']]
                inp = data['inputs']
                geo = ep_eval._img_geo(metas, inp['imgs'])
                feats_2d = ep_eval._img_feats_2d(inp['imgs'])
                F_C = ep_eval._img_bev(feats_2d, inp['points'], geo)
                F_L = ep_eval._pts_bev(inp['points'])
                mu = ep_eval.poe_fuser(F_C, F_L)['mu_F'].float()
                ft, stages = teacher_fuser_forward(
                    ep_eval.fusion_layer, F_C, F_L)
                # 后端诊断只复用两路 BEV，及时释放无关图像/预处理临时量。
                del data, inp, geo, metas, feats_2d
                original_ok, detail = compare_tensor('poe_vs_teacher', mu, ft)
                original_finite = all(bool(torch.isfinite(t).all()) for t in (
                    F_C, F_L, mu, ft, stages['conv'], stages['bn']))
                native_ok = None
                precision_detail = 'fp64_diag skipped (FP32 gate passed)'
                backend_detail = 'backend_diag skipped (FP32 gate passed)'
                if not original_ok or selected_backend == 'no_cudnn':
                    bad_coords = (~torch.isclose(
                        mu, ft, rtol=1e-4, atol=1e-5)).nonzero().cpu()
                    if not original_ok:
                        try:
                            precision_detail = poe_fp64_diagnostic(
                                ep_eval, F_C, F_L, mu, ft, stages=stages,
                                bad_coords=bad_coords)
                        except Exception as e:
                            if selected_backend == 'no_cudnn':
                                raise
                            precision_detail = 'fp64_diag unavailable: %r' % e
                    del stages
                    try:
                        native_ok, backend_detail = fuser_backend_diagnostic(
                            ep_eval, F_C, F_L, mu, ft, bad_coords,
                            diagnostic_only=(selected_backend == 'original'))
                    except Exception as e:
                        if selected_backend == 'no_cudnn':
                            raise
                        backend_detail = 'backend_diag unavailable: %r' % e
        finally:
            ep_pp.force_mode = old_mode
            model.train(was_training)
        # 只允许显式选择受控闸门；原后端非有限值不得被受控 PASS 掩盖。
        ok = original_finite and (
            original_ok if selected_backend == 'original' else native_ok)
        warning = ('; WARNING original backend tolerance check failed'
                   if ok and not original_ok else '')
        return ok, (
            'initialized=%s strict_fp32=True selected_backend=%s '
            'original_ok=%s original_finite=%s no_cudnn_ok=%s%s; '
            '%s; %s\n    %s'
            % (ok_init, selected_backend, original_ok, original_finite,
               native_ok, warning, detail, precision_detail, backend_detail))
    check(report, 'i_teacher_equivalence', _i)

    # (c) pipeline 无 ObjectSample
    def _c():
        ds = copy.deepcopy(cfg.train_dataloader['dataset'])
        if ds.get('type') == 'CBGSDataset':
            ds = ds['dataset']
        types = [t.get('type') for t in ds.get('pipeline', [])]
        return ('ObjectSample' not in types), 'pipeline=%s' % types
    check(report, 'c_no_object_sample', _c)

    # (a)+(b) 三模式 loss 键齐全 + 初始 Λ≈1
    losses_by_mode = {}

    @torch.no_grad()
    def _run_all():
        msgs = []
        for m in ('clean', 'corrupt_cam', 'corrupt_lidar'):
            losses, loss_sum, log_vars = run_mode(model, batch, m)
            # 只保留无计算图的 CPU 日志；backward 另跑独立前向。
            losses_by_mode[m] = {
                k: ([v.detach().cpu() for v in value]
                    if isinstance(value, list) else value.detach().cpu())
                for k, value in losses.items()
            }
            msgs.append(
                '%s loss_sum=%.6g log_vars=%d'
                % (m, float(loss_sum.detach()), len(log_vars)))
            del losses, loss_sum, log_vars
        return True, '; '.join(msgs)
    check(report, 'run_three_modes', _run_all)

    def _a():
        miss = {}
        for m, ls in losses_by_mode.items():
            need = LAMBDA_KEYS + ['loss_teach', 'teach_nll_raw'] + WARN_KEYS
            miss[m] = [k for k in need if k not in ls]
        bad = {m: v for m, v in miss.items() if v}
        return (len(bad) == 0), 'missing=%s' % (bad or 'none')
    check(report, 'a_keys_present', _a)

    def _b():
        ls = losses_by_mode['clean']
        mc = float(ls['lambda_C_clean_sum'] / ls['lambda_C_clean_cnt'])
        ml = float(ls['lambda_L_clean_sum'] / ls['lambda_L_clean_cnt'])
        ok = abs(mc - 1.0) < 1e-6 and abs(ml - 1.0) < 1e-6
        return ok, 'mean(Lam_C)=%.6f mean(Lam_L)=%.6f' % (mc, ml)
    check(report, 'b_lambda_init_one', _b)
    losses_by_mode.clear()

    # (f) backward 冒烟
    def _f():
        _, total, _ = run_mode(model, batch, 'clean')
        model.zero_grad(set_to_none=True)
        train_no_grad, frozen_has_grad = [], []
        try:
            total.backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is None:
                    train_no_grad.append(n)
                if (not p.requires_grad) and (p.grad is not None):
                    frozen_has_grad.append(n)
        finally:
            model.zero_grad(set_to_none=True)
        ok = (len(train_no_grad) == 0 and len(frozen_has_grad) == 0)
        return ok, ('train_no_grad=%s frozen_has_grad=%s'
                    % (train_no_grad[:5], frozen_has_grad[:5]))
    check(report, 'f_backward_smoke', _f)

    # (g) 整路置零强制路径冒烟
    @torch.no_grad()
    def _g():
        msgs = []
        all_ok = True
        for fm, fc in (('corrupt_cam', 'zero_image'),
                       ('corrupt_lidar', 'zero_points')):
            model.data_preprocessor.force_mode = fm
            model.data_preprocessor.force_corruption = fc
            try:
                data = model.data_preprocessor(copy.deepcopy(batch), True)
                ls = model.loss(data['inputs'], data['data_samples'])
                need = [
                    'lambda_C_%s_sum' % fm,
                    'lambda_C_%s_cnt' % fm,
                    'lambda_L_%s_sum' % fm,
                    'lambda_L_%s_cnt' % fm,
                    'loss_teach',
                    'teach_nll_raw',
                ]
                missing = [k for k in need if k not in ls]
                all_ok = all_ok and not missing
                msgs.append('%s/%s missing=%s'
                            % (fm, fc, missing or 'none'))
                del data, ls
            finally:
                model.data_preprocessor.force_mode = None
                model.data_preprocessor.force_corruption = None
        return all_ok, '; '.join(msgs)
    check(report, 'g_zero_paths_smoke', _g)

    # (h) scheduler 初始化状态 + param_scheduler epoch 端点断言
    def _h():
        from mmengine.optim import build_optim_wrapper
        from mmengine.registry import PARAM_SCHEDULERS
        ow = build_optim_wrapper(model, copy.deepcopy(cfg.optim_wrapper))
        base_lrs = sorted({g['lr'] for g in ow.optimizer.param_groups})
        scheduler_cfgs = copy.deepcopy(cfg.get('param_scheduler', []))
        max_ep = cfg.train_cfg.get('max_epochs')
        out_of_bounds = []
        static = []
        for idx, scheduler_cfg in enumerate(scheduler_cfgs):
            scheduler_type = scheduler_cfg.get('type')
            static.append((
                scheduler_type, scheduler_cfg.get('T_max'),
                scheduler_cfg.get('end'), scheduler_cfg.get('by_epoch', True)))
            if (scheduler_cfg.get('T_max') is not None
                    and scheduler_cfg['T_max'] > max_ep):
                out_of_bounds.append(
                    '#%d %s T_max=%s' %
                    (idx, scheduler_type, scheduler_cfg['T_max']))
            if (scheduler_cfg.get('by_epoch', True)
                    and scheduler_cfg.get('end') is not None
                    and scheduler_cfg['end'] > max_ep):
                out_of_bounds.append(
                    '#%d %s end=%s' %
                    (idx, scheduler_type, scheduler_cfg['end']))

        initialized = '无法构造，仅打印静态配置'
        try:
            schedulers = [
                PARAM_SCHEDULERS.build(
                    scheduler_cfg,
                    default_args=dict(
                        optimizer=ow, epoch_length=epoch_length))
                for scheduler_cfg in scheduler_cfgs
            ]
            initial_lrs = [group['lr']
                           for group in ow.optimizer.param_groups]
            initial_momenta = []
            for group in ow.optimizer.param_groups:
                if 'momentum' in group:
                    initial_momenta.append(group['momentum'])
                elif 'betas' in group:
                    initial_momenta.append(group['betas'][0])
                else:
                    initial_momenta.append(None)
            # 构造器已经执行初始化 step；此处不得在 optimizer.step() 前再推进。
            initialized = 'lr_unique=%s momentum_unique=%s groups=%d' % (
                list(dict.fromkeys(initial_lrs)),
                list(dict.fromkeys(initial_momenta)), len(initial_lrs))
        except Exception as e:
            initialized += ' (%r)' % e

        ok = not out_of_bounds
        return ok, (
            'base_lrs=%s initialized=(%s) max_epochs=%s scheduler=%s '
            'out_of_bounds=%s'
            % (base_lrs, initialized, max_ep, static,
               out_of_bounds or 'none'))
    check(report, 'h_optim_scheduler', _h)

    # (d) 教师路径受控 allclose（EP clean F_T vs 原 BEVFusion fusion_layer 输出）
    def _d():
        base_cfg_path = args.base_config
        if base_cfg_path is None:
            # 从 EP config 的 _base_ 推断（4xA30 BEVFusion config）
            base_cfg_path = os.path.join(
                REPO_ROOT, 'projects/BEVFusion/configs/'
                'bevfusion_lidar-cam_voxel0075_4xa30-amp-accum_nus-3d.py')
        _, stock = build_model_from_cfg(base_cfg_path)
        load_ckpt_loose(stock, args.checkpoint)
        state_ok, state_msg = compare_teacher_state(model, stock)
        print('  [d diagnostic] ' + state_msg)
        stock = stock.to(args.device).eval()
        was_training = model.training
        model.eval()
        try:
            with strict_fp32(deterministic=True):
                stock_first = teacher_snapshot(stock, batch)
                stock_repeat = teacher_snapshot(stock, batch)
                repeat_ok, repeat_msg = compare_snapshots(
                    'stock_self_repeat', stock_repeat, stock_first)
                print('  [d diagnostic] ' + repeat_msg)
                del stock_repeat
                ep = teacher_snapshot(model, batch, ep_path=True)
                path_ok, path_msg = compare_snapshots(
                    'ep_vs_stock', ep, stock_first)
        finally:
            model.train(was_training)
        return (state_ok and repeat_ok and path_ok), (
            'strict_fp32=True deterministic=True cublas_workspace=%s; '
            '%s\n    %s\n    %s'
            % (os.environ.get('CUBLAS_WORKSPACE_CONFIG'), state_msg,
               repeat_msg, path_msg))
    check(report, 'd_teacher_allclose', _d)

    # (e) R0 权重拷贝断言
    def _e():
        if not args.r0_config:
            return True, 'SKIP（未提供 --r0-config）'
        from projects.EPFusion.epfusion.hooks import copy_weights
        _, r0 = build_model_from_cfg(
            args.r0_config,
            cfg_options=args.cfg_options,
            data_root=args.data_root)
        actual = (
            r0.fusion_mode,
            r0.w_teach,
            r0.data_preprocessor.emit_clean)
        config_ok = (
            r0.fusion_mode == 'convfuser'
            and r0.w_teach == 0.0
            and r0.data_preprocessor.emit_clean is False)
        if not config_ok:
            return False, (
                'R0 config invalid: fusion_mode=%r w_teach=%r '
                'emit_clean=%r' % actual)
        load_ckpt_loose(r0, args.checkpoint)
        ok_copy = copy_weights(r0)  # 显式调用（脚本不经 Runner.train）
        eq = True
        sd_s = r0.student_fuser.state_dict()
        sd_t = r0.fusion_layer.state_dict()
        for k in sd_s:
            if not torch.equal(sd_s[k], sd_t[k]):
                eq = False
                break
        return (ok_copy and eq), (
            'fusion_mode=%r w_teach=%r emit_clean=%r copied=%s '
            'all_equal=%s' % (actual + (ok_copy, eq)))
    check(report, 'e_r0_weight_copy', _e)

    # ---- 总览 ----
    print('\n' + '=' * 70 + '\n断言总览\n' + '=' * 70)
    n_fail = 0
    for name, status, detail in report:
        if status != 'PASS':
            n_fail += 1
        print('  [%-5s] %-22s %s' % (status, name, detail))
    print('\n%d/%d 通过；报告写入 %s'
          % (len(report) - n_fail, len(report), os.path.abspath(out)))
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == '__main__':
    main()
