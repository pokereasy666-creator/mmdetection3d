#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""m0_probe.py — EP-Fusion M0 运行时探针（只读、无副作用）。

定位（CLAUDE.md 铁律 13/14/15/16/17/18）：
  把所有"需要 GPU / nuScenes 数据 / 训练日志 / 已安装环境才能回答"的勘察问题
  打包成一次性探针；全部结论写入单一文本报告，由用户拷回。
  - 运行前自检：检测到任何训练进程（train.py / torchrun / GPU compute 进程）即拒绝执行；
  - 只读：不写 work_dir、不动 checkpoint、不落任何中间文件（除报告本身）；
    P4 的反传只作用于脚本内临时优化器与内存中的权重副本，不保存；
  - 零新依赖：仅用 torch / mmengine / mmcv / mmdet / mmdet3d / numpy / 标准库；
  - 不调 git：版本信息读仓库根 VERSION 文件，读不到记 "unknown"；
  - 一切可调项走 CLI（默认值即 4×A30 复现 config）。

探针内容：
  P1 环境实测：五库版本与 __file__（import 路径自检）、CUDA/GPU 型号显存、
     BEVFusion 自定义算子可导入性、完整 pip list（铁律 15 依赖白名单）。
  P2 复现产物：work_dir 训练日志中最近/最终 val 指标与 loss 曲线尾部；
     官方 / 复现 checkpoint 各自加载兼容性（missing / unexpected keys）。
  P3 形状实测：load checkpoint 前向一个真实 val batch，hook 打印
     相机 BEV、LiDAR BEV、fusion 输入输出的真实张量形状。
  P4 显存/走时实测：batch=1/2 × AMP 开/关，各 dry-run 数个 iteration
     （含反传；只训 pts_backbone+pts_neck+bbox_head 参数集），记录峰值显存与单 iter 走时。
  P5 数据管线实测：1 个样本过完整 train pipeline 后的 keys 与各张量形状/值域、
     点云第 5 维语义线索（ring index vs 时间戳）、aug 矩阵存在性。
  P6 mmengine 行为验证：model.train() 调用时机（EpochBasedTrainLoop / ValLoop 源码）、
     Runner.train 中 load_or_resume 先于 before_train 的顺序（R0WeightCopyHook 依据）、
     OptimWrapper 对 requires_grad=False 参数的处理（冻结参数位级不变断言）、
     BaseModel.train_step 调用 self.parse_losses 的证据。

服务器执行（务必等复现训练结束后再跑）：
  bash scripts/deploy.sh --data <nuscenes> --work-dirs <work_dirs> --ops-from <旧部署目录>
  bash scripts/m0_probe.sh \
      --config projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_4xa30-amp-accum_nus-3d.py \
      --repro-ckpt work_dirs/baseline1/epoch_19.pth \
      --official-ckpt <path>/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth \
      --work-dir work_dirs/<run>
"""
import argparse
import copy
import datetime
import glob
import importlib
import inspect
import json
import os
import platform
import subprocess
import sys
import time
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)  # 与 tools/dist_train.sh 的 PYTHONPATH 约定一致

DEFAULT_CONFIG = ('projects/BEVFusion/configs/'
                  'bevfusion_lidar-cam_voxel0075_4xa30-amp-accum_nus-3d.py')


# ---------------------------------------------------------------- 基础设施
class Tee(object):
    """stdout 同时写入报告文件（铁律 18：单文件报告）。"""

    def __init__(self, path):
        self.file = open(path, 'w', encoding='utf-8')
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def section(title):
    print('\n' + '=' * 78)
    print(title)
    print('=' * 78)


def import_first(module_paths, attr):
    """从多个候选模块路径里取第一个能拿到 attr 的（防 mmengine 版本间模块路径漂移）。

    本机无 mmengine 源，无法静态确认 EpochBasedTrainLoop/ValLoop 究竟从
    `mmengine.runner` 还是 `mmengine.runner.loops` 暴露——两条路径都试，
    谁先成功用谁；都失败则抛出最后一个异常（由 P6 的 try/except 兜住）。
    """
    last_err = None
    for mp in module_paths:
        try:
            return getattr(importlib.import_module(mp), attr)
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise ImportError('无法从 %s 导入 %s（最后错误 %r）'
                      % (module_paths, attr, last_err))


def read_version():
    """铁律 16：不调 git，读 VERSION；读不到记 unknown，不许报错。"""
    try:
        with open(os.path.join(REPO_ROOT, 'VERSION'), encoding='utf-8') as f:
            return f.read().strip()
    except Exception:
        return 'unknown'


def detect_training_processes():
    """铁律 13：检测训练进程。返回可疑进程描述列表（空 = 安全）。"""
    suspects = []
    try:
        out = subprocess.run(['ps', '-eo', 'pid,args'], capture_output=True,
                             text=True, timeout=15).stdout
        for line in out.splitlines():
            low = line.lower()
            if 'm0_probe' in low:
                continue  # 本探针自身
            if any(k in low for k in ('tools/train.py', 'torchrun',
                                      'torch.distributed.launch',
                                      'torch.distributed.run', 'dist_train')):
                suspects.append('[ps] ' + line.strip()[:200])
    except Exception as e:
        suspects.append('[ps] 进程扫描失败（按"无法确认"处理，拒绝执行）: %r' % e)
    try:
        r = subprocess.run(
            ['nvidia-smi',
             '--query-compute-apps=pid,process_name,used_memory',
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=15)
        if r.returncode == 0:
            for line in r.stdout.strip().splitlines():
                if line.strip():
                    suspects.append('[gpu] compute 进程: ' + line.strip())
    except FileNotFoundError:
        pass  # 无 nvidia-smi：仅凭 ps 判断
    except Exception as e:
        suspects.append('[gpu] nvidia-smi 查询失败（按"无法确认"处理）: %r' % e)
    return suspects


def shape_of(x):
    """递归打印 tensor 结构形状。"""
    import torch
    if isinstance(x, torch.Tensor):
        return 'Tensor%s %s' % (tuple(x.shape), x.dtype)
    if isinstance(x, (list, tuple)):
        if len(x) > 6:
            return '%s[len=%d, first=%s]' % (type(x).__name__, len(x),
                                             shape_of(x[0]))
        return [shape_of(i) for i in x]
    if isinstance(x, dict):
        return {k: shape_of(v) for k, v in x.items()}
    return type(x).__name__


def build_model_from_cfg(config_path):
    """解析 config（触发 custom_imports）并构建模型（不加载权重）。"""
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    cfg = Config.fromfile(config_path)
    ci = cfg.get('custom_imports', None)
    if ci:
        from mmengine.utils import import_modules_from_strings
        import_modules_from_strings(**ci)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    from mmdet3d.registry import MODELS
    # BEVFusion.__init__ 会 pop data_preprocessor.voxelize_cfg，deepcopy 保 cfg 可复用
    model = MODELS.build(copy.deepcopy(cfg.model))
    return cfg, model


def load_ckpt_report(model, path, tag):
    """以 strict=False 加载 ckpt 并报告兼容性（不修改磁盘）。"""
    import torch
    print('\n--- checkpoint 兼容性: %s (%s)' % (tag, path))
    if not path or not os.path.isfile(path):
        print('    跳过：路径未提供或文件不存在')
        return False
    ckpt = torch.load(path, map_location='cpu')
    sd = ckpt.get('state_dict', ckpt)
    meta = ckpt.get('meta', {})
    if meta:
        keys = sorted(meta.keys())
        print('    meta keys: %s' % keys)
        for k in ('mmdet3d_version', 'epoch', 'iter', 'seed'):
            if k in meta:
                print('    meta[%s] = %s' % (k, meta[k]))
    sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
    res = model.load_state_dict(sd, strict=False)
    print('    state_dict 张量数: %d' % len(sd))
    print('    missing keys   : %d  %s' % (len(res.missing_keys),
                                           res.missing_keys[:20]))
    print('    unexpected keys: %d  %s' % (len(res.unexpected_keys),
                                           res.unexpected_keys[:20]))
    return True


# ---------------------------------------------------------------- P1
def probe_p1(args):
    section('P1 环境实测（版本 / GPU / import 路径 / 算子 / pip list 白名单）')
    print('python   : %s' % sys.version.replace('\n', ' '))
    print('platform : %s' % platform.platform())
    print('hostname : %s' % platform.node())
    mods = {}
    for name in ('torch', 'mmengine', 'mmcv', 'mmdet', 'mmdet3d', 'numpy'):
        try:
            m = importlib.import_module(name)
            mods[name] = m
            print('%-9s: %-14s %s' % (name, getattr(m, '__version__', '?'),
                                      getattr(m, '__file__', '?')))
        except Exception as e:
            print('%-9s: 导入失败 %r' % (name, e))
    # import 路径自检：mmdet3d 必须来自当前解压树（zip 部署关键）
    if 'mmdet3d' in mods:
        f = os.path.abspath(getattr(mods['mmdet3d'], '__file__', ''))
        if f.startswith(REPO_ROOT + os.sep):
            print('[自检] mmdet3d 来自当前仓库树: OK')
        else:
            print('[自检] 警告：mmdet3d 不来自当前树（%s）——' % f)
            print('       请经 scripts/m0_probe.sh 启动（它设置 PYTHONPATH=仓库根）')
    if 'torch' in mods:
        torch = mods['torch']
        try:
            cudnn_v = torch.backends.cudnn.version()
        except Exception as e:  # cudnn 未编译/不可用时不应连累整个 P1
            cudnn_v = 'N/A(%r)' % e
        print('torch.version.cuda = %s, cudnn = %s' %
              (torch.version.cuda, cudnn_v))
        print('CUDA available = %s, device_count = %d' %
              (torch.cuda.is_available(), torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print('  GPU%d: %s, %.1f GB, sm_%d%d' %
                  (i, p.name, p.total_memory / 1024 ** 3, p.major, p.minor))
    # BEVFusion 自定义算子
    for ext in ('projects.BEVFusion.bevfusion.ops.voxel.voxel_layer',
                'projects.BEVFusion.bevfusion.ops.bev_pool.bev_pool_ext'):
        try:
            importlib.import_module(ext)
            print('[算子] %s : OK' % ext)
        except Exception as e:
            print('[算子] %s : 导入失败 %r' % (ext, e))
            print('       （fresh zip 需先 bash scripts/deploy.sh --ops-from <旧目录>）')
    # 完整 pip list（铁律 15 白名单；用 importlib.metadata，不依赖 pip 可执行）
    print('\n[pip list]（铁律 15：后续新代码 import 以此为白名单）')
    try:
        import importlib.metadata as md
        rows = []
        for d in md.distributions():
            try:
                rows.append((d.metadata['Name'] or '?', d.version or '?'))
            except Exception:
                pass
        for name, ver in sorted(rows, key=lambda r: r[0].lower()):
            print('  %-40s %s' % (name, ver))
        print('  （共 %d 个发行版）' % len(rows))
    except Exception as e:
        print('  importlib.metadata 枚举失败 %r' % e)


# ---------------------------------------------------------------- P2
def probe_p2(args):
    section('P2 复现产物（val 指标 / loss 尾部 / checkpoint 兼容性）')
    if args.work_dir and os.path.isdir(args.work_dir):
        jsons = sorted(
            glob.glob(os.path.join(args.work_dir, '*', 'vis_data', '*.json')),
            key=os.path.getmtime)
        logs = sorted(glob.glob(os.path.join(args.work_dir, '*', '*.log')),
                      key=os.path.getmtime)
        val_rows, loss_rows = [], []
        # ---- 首选：解析 vis_data 下的 JSON-lines scalars ----
        if jsons:
            latest = jsons[-1]
            print('最新 scalars 文件: %s' % latest)
            with open(latest, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue  # 非 JSON-lines 行跳过（兼容非标准格式）
                    if any(('NDS' in k or 'mAP' in k) for k in d):
                        val_rows.append(d)
                    elif any('loss' in k for k in d):
                        loss_rows.append(d)
            print('[val 指标记录数: %d；最近 3 条]' % len(val_rows))
            for d in val_rows[-3:]:
                print('  %s' % json.dumps(d, ensure_ascii=False))
            if loss_rows:
                print('[train loss 尾部 10 条]')
                for d in loss_rows[-10:]:
                    brief = {k: v for k, v in d.items()
                             if 'loss' in k or k in ('epoch', 'iter', 'lr')}
                    print('  %s' % json.dumps(brief, ensure_ascii=False))
        else:
            print('未在 %s/*/vis_data/ 下找到 scalars json' % args.work_dir)

        # ---- 关键数（复现 NDS/mAP，决定 Δ>0.5 排查门）：vis_data 读不出则
        #      回退到 .log 文本 grep，绝不因日志格式而读不出最关键的数 ----
        if val_rows:
            last = val_rows[-1]
            nds = [v for k, v in last.items() if k.endswith('NDS')]
            mAP = [v for k, v in last.items() if k.endswith('mAP')]
            print('\n>>> 复现最终 val（来自 vis_data）: NDS=%s mAP=%s' % (nds, mAP))
        elif logs:
            print('\n>>> vis_data 未解析到 val 指标，回退 .log grep NDS/mAP:')
            hit = []
            for lg in logs:
                with open(lg, encoding='utf-8', errors='replace') as f:
                    for line in f:
                        if 'NDS' in line or 'mAP' in line:
                            hit.append(line.rstrip())
            for line in hit[-12:]:  # 尾部若干条≈最终评测结果
                print('  ' + line)
            if not hit:
                print('  .log 中也未出现 NDS/mAP（训练可能尚未跑到 val）')
        else:
            print('\n>>> 既无 vis_data val 行，也无 .log 文件，无法取复现 NDS')
        print('>>> 官方参照（projects/BEVFusion/README.md:71）: '
              'NDS 71.4 / mAP 68.6；|Δ| > 0.5 NDS 须排查后再继续 M0')

        # ---- 最新 .log 尾部，便于人工核对训练收尾 ----
        if logs:
            print('\n[最新 .log 尾部 20 行] %s' % logs[-1])
            with open(logs[-1], encoding='utf-8', errors='replace') as f:
                for line in f.readlines()[-20:]:
                    print('  ' + line.rstrip())
    else:
        print('--work-dir 未提供或不存在，跳过日志解析')

    # checkpoint 兼容性（CPU 上构建模型即可）
    try:
        _, model = build_model_from_cfg(args.config)
        n_total = sum(p.numel() for p in model.parameters())
        print('\n模型构建 OK（config=%s），参数量 %.2f M' %
              (args.config, n_total / 1e6))
        load_ckpt_report(model, args.official_ckpt, 'official')
        load_ckpt_report(model, args.repro_ckpt, 'repro')
        del model
    except Exception:
        print('模型构建/加载失败：\n%s' % traceback.format_exc())


# ---------------------------------------------------------------- P3
def probe_p3(args):
    section('P3 形状实测（相机 BEV / LiDAR BEV / fusion 输入输出）')
    import torch
    if not torch.cuda.is_available():
        print('无 CUDA，跳过 P3')
        return
    cfg, model = build_model_from_cfg(args.config)
    ckpt = args.repro_ckpt if (args.repro_ckpt and os.path.isfile(
        args.repro_ckpt)) else args.official_ckpt
    if ckpt and os.path.isfile(ckpt):
        sd = torch.load(ckpt, map_location='cpu')
        sd = sd.get('state_dict', sd)
        sd = {(k[7:] if k.startswith('module.') else k): v
              for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        print('已加载 ckpt: %s' % ckpt)
    else:
        print('警告：无可用 ckpt，用随机初始化前向（形状结论不受影响）')
    model = model.to(args.device).eval()

    from mmengine.runner import Runner
    dl_cfg = copy.deepcopy(cfg.val_dataloader)
    dl_cfg['num_workers'] = 2
    dl_cfg['persistent_workers'] = False
    loader = Runner.build_dataloader(dl_cfg)
    batch = next(iter(loader))  # DataLoader 的 collate_fn(pseudo_collate) 已 collate

    # 先实测属性名（不凭记忆挂 hook）；单个属性名错也不应崩掉整个 P3
    print('model.named_children(): %s' % [n for n, _ in model.named_children()])

    records = []

    def mk_hook(name):
        def hook(mod, inputs, output):
            records.append((name, shape_of(list(inputs)), shape_of(output)))
        return hook

    hook_targets = [
        ('view_transform', '相机 BEV 出口'),
        ('pts_middle_encoder', 'LiDAR BEV 出口'),
        ('fusion_layer', 'ConvFuser'),
        ('pts_backbone', 'SECOND'),
        ('pts_neck', 'SECONDFPN'),
    ]
    handles = []
    for attr, desc in hook_targets:
        mod = getattr(model, attr, None)
        if mod is None:
            print('  [跳过 hook] model.%s 不存在' % attr)
            continue
        handles.append(
            mod.register_forward_hook(mk_hook('%s(%s)' % (attr, desc))))
    try:
        # 走官方推理入口 test_step（内部 data_preprocessor + predict 前向），
        # 不手搓 data_preprocessor/extract_feat 绕过 BEVFusion 的 forward 契约
        with torch.no_grad():
            model.test_step(batch)
    finally:
        for h in handles:
            h.remove()
    for name, fin, fout in records:
        print('\n[%s]' % name)
        print('  inputs : %s' % fin)
        print('  output : %s' % fout)
    print('\n核对要点：fusion_layer inputs 应为 [相机 BEV(80ch), LiDAR BEV(256ch)]'
          '（bevfusion.py:271/273 的 append 顺序：img 在前）；'
          '输出应为 256ch；空间均应为 180×180。')
    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- P4
def probe_p4(args):
    section('P4 显存/走时实测（batch×AMP 矩阵；只训 BEV encoder + 检测头）')
    import torch
    if not torch.cuda.is_available():
        print('无 CUDA，跳过 P4')
        return
    trainable_prefixes = ('pts_backbone', 'pts_neck', 'bbox_head')
    frozen_eval_attrs = ('img_backbone', 'img_neck', 'view_transform',
                         'pts_voxel_encoder', 'pts_middle_encoder',
                         'fusion_layer')
    batch_sizes = [int(b) for b in args.batch_sizes.split(',') if b.strip()]
    for bs in batch_sizes:
        for amp in (True, False):
            tag = 'batch=%d amp=%s' % (bs, amp)
            try:
                cfg, model = build_model_from_cfg(args.config)
                ckpt = args.repro_ckpt if (args.repro_ckpt and os.path.isfile(
                    args.repro_ckpt)) else args.official_ckpt
                if ckpt and os.path.isfile(ckpt):
                    sd = torch.load(ckpt, map_location='cpu')
                    sd = sd.get('state_dict', sd)
                    sd = {(k[7:] if k.startswith('module.') else k): v
                          for k, v in sd.items()}
                    model.load_state_dict(sd, strict=False)
                model = model.to(args.device)
                # 模拟 M0 可训练集合（铁律 1/10 的探针级模拟）
                n_train = n_frozen = 0
                for name, p in model.named_parameters():
                    t = name.startswith(trainable_prefixes)
                    p.requires_grad_(t)
                    n_train += p.numel() * t
                    n_frozen += p.numel() * (not t)
                model.train()
                for a in frozen_eval_attrs:
                    m = getattr(model, a, None)
                    if m is not None:
                        m.eval()
                print('\n[%s] 可训练 %.2fM / 冻结 %.2fM 参数' %
                      (tag, n_train / 1e6, n_frozen / 1e6))

                from mmengine.runner import Runner
                from mmengine.optim import build_optim_wrapper
                dl_cfg = copy.deepcopy(cfg.train_dataloader)
                ds = dl_cfg['dataset']
                if ds.get('type') == 'CBGSDataset':
                    ds = ds['dataset']  # 跳过 CBGS 包装，省构建时间（pipeline 相同）
                dl_cfg['dataset'] = ds
                dl_cfg['batch_size'] = bs
                dl_cfg['num_workers'] = 2
                dl_cfg['persistent_workers'] = False
                loader = Runner.build_dataloader(dl_cfg)
                batch = next(iter(loader))

                # 官方入口：amp=True→AmpOptimWrapper(loss_scale='dynamic')，
                # amp=False→OptimWrapper；model.train_step 内部完成
                # preprocess+loss+parse+backward+step，对齐真实 --amp 路径。
                # 两套 AMP 机制不可叠加，故不再手搓 GradScaler/autocast。
                ow_cfg = dict(
                    type='AmpOptimWrapper' if amp else 'OptimWrapper',
                    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.01))
                if amp:
                    ow_cfg['loss_scale'] = 'dynamic'
                optim_wrapper = build_optim_wrapper(model, ow_cfg)

                iters = max(2, args.timing_iters)
                times = []
                last_log = None
                torch.cuda.reset_peak_memory_stats()
                for it in range(iters):
                    torch.cuda.synchronize()
                    t0 = time.time()
                    last_log = model.train_step(batch, optim_wrapper)
                    torch.cuda.synchronize()
                    times.append(time.time() - t0)
                peak_alloc = torch.cuda.max_memory_allocated() / 1024 ** 3
                peak_resv = torch.cuda.max_memory_reserved() / 1024 ** 3
                print('  峰值显存 allocated=%.2f GB reserved=%.2f GB' %
                      (peak_alloc, peak_resv))
                print('  iter 走时: %s（首 iter 含 warmup；稳态≈%.2fs）' %
                      (['%.2fs' % t for t in times], times[-1]))
                if isinstance(last_log, dict) and 'loss' in last_log:
                    print('  train_step log loss=%s（仅证通路打通）'
                          % last_log.get('loss'))
                del model, optim_wrapper, loader, batch
                torch.cuda.empty_cache()
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print('  [%s] OOM: %s' % (tag, str(e)[:200]))
                    torch.cuda.empty_cache()
                else:
                    print('  [%s] 失败:\n%s' % (tag, traceback.format_exc()))
            except Exception:
                print('  [%s] 失败:\n%s' % (tag, traceback.format_exc()))


# ---------------------------------------------------------------- P5
def probe_p5(args):
    section('P5 数据管线实测（train pipeline 输出结构）')
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    cfg = Config.fromfile(args.config)
    ci = cfg.get('custom_imports', None)
    if ci:
        from mmengine.utils import import_modules_from_strings
        import_modules_from_strings(**ci)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    ds_cfg = copy.deepcopy(cfg.train_dataloader['dataset'])
    if ds_cfg.get('type') == 'CBGSDataset':
        ds_cfg = ds_cfg['dataset']
    print('train pipeline transform 顺序（config 解析值）:')
    for i, t in enumerate(ds_cfg.get('pipeline', [])):
        print('  %2d. %s' % (i, t.get('type')))
        assert t.get('type') != 'ObjectSample', \
            'GT-sampling 出现在 train pipeline！与静态勘察矛盾，停止'
    print('  [断言通过] pipeline 中无 ObjectSample（GT-sampling 关闭，铁律 4）')

    from mmdet3d.registry import DATASETS
    dataset = DATASETS.build(ds_cfg)
    sample = dataset[0]
    print('\n样本顶层 keys: %s' % list(sample.keys()))
    inputs = sample['inputs']
    print('inputs keys: %s' % list(inputs.keys()))
    img = inputs.get('img')
    if img is not None:
        print('inputs[img]   : %s  min=%.2f max=%.2f（归一化前应为 0-255 量级）'
              % (shape_of(img), float(img.min()), float(img.max())))
    pts = inputs.get('points')
    if pts is not None:
        print('inputs[points]: %s' % shape_of(pts))
        for d in range(pts.shape[1]):
            col = pts[:, d]
            print('  dim%d: min=%.3f max=%.3f mean=%.3f' %
                  (d, float(col.min()), float(col.max()), float(col.mean())))
        if pts.shape[1] >= 5:
            col = pts[:, 4]
            uniq = col.unique()
            intlike = bool(((col - col.round()).abs() < 1e-6).float()
                           .mean() > 0.999)
            # 阈值放宽到 128：nuScenes 单帧 ring∈[0,31]，但 9-sweep 叠加
            # （及不同采集）会抬高唯一整数值的边界，用 128 作宽松上界，
            # 避免把 ring index 误判为连续量（时间戳）。
            ring_like = intlike and uniq.numel() <= 128
            print('  [第 5 维语义线索] unique=%d, 整数性=%s → %s' %
                  (uniq.numel(), intlike,
                   'ring index 可能性高（抽线损坏可直接用）'
                   if ring_like else
                   '疑似时间戳/连续量（抽线损坏须回退俯仰角分箱）'))
    ds_sample = sample.get('data_samples')
    if ds_sample is not None:
        meta = ds_sample.metainfo
        print('\nmetainfo keys: %s' % sorted(meta.keys()))
        for k in ('img_aug_matrix', 'lidar_aug_matrix',
                  'transformation_3d_flow', 'num_pts_feats'):
            print('  %s: %s' % (k, '存在' if k in meta else '缺失'))
        print('gt_instances_3d: %s' %
              ('存在(%d 个框)' % len(ds_sample.gt_instances_3d)
               if hasattr(ds_sample, 'gt_instances_3d') else '缺失'))


# ---------------------------------------------------------------- P6
def probe_p6(args):
    section('P6 mmengine 行为验证（train() 时机 / hook 顺序 / 优化器 / train_step）')
    import torch
    from torch import nn

    print('[6.1] EpochBasedTrainLoop.run / run_epoch 源码（model.train() 调用时机）:')
    # 本机无 mmengine 源，类暴露路径无法静态确认；两条候选路径都试（见 import_first）。
    EpochBasedTrainLoop = import_first(
        ['mmengine.runner', 'mmengine.runner.loops'], 'EpochBasedTrainLoop')
    ValLoop = import_first(
        ['mmengine.runner', 'mmengine.runner.loops'], 'ValLoop')
    print(inspect.getsource(EpochBasedTrainLoop.run))
    print(inspect.getsource(EpochBasedTrainLoop.run_epoch))
    print('[6.2] ValLoop.run 源码（eval/train 切换时机）:')
    print(inspect.getsource(ValLoop.run))

    print('[6.3] Runner.train 中 load_or_resume 与 train_loop.run 的先后'
          '（R0WeightCopyHook before_train 时机依据；机器判定仅辅助，主看行号人工确认）:')
    from mmengine.runner import Runner
    src_lines = inspect.getsource(Runner.train).splitlines()

    def find_call_lineno(needle):
        """needle 首次作为【实际调用语句】（非注释行、含 self. 前缀）出现的行号。"""
        for idx, ln in enumerate(src_lines):
            if ln.strip().startswith('#'):     # 整行注释跳过
                continue
            code = ln.split('#', 1)[0]         # 去掉行内注释
            if needle in code and 'self.' in code:
                return idx
        return -1

    # 打印所有相关行（含行号）供人工确认——机器判定不作唯一依据
    for idx, ln in enumerate(src_lines):
        if any(k in ln for k in ('load_or_resume', 'train_loop.run',
                                 'call_hook')):
            print('  L%03d: %s' % (idx, ln.rstrip()))
    li_load = find_call_lineno('load_or_resume')
    li_run = find_call_lineno('train_loop.run')
    print('  [实际调用行号] load_or_resume=L%d, train_loop.run=L%d' %
          (li_load, li_run))
    if li_load >= 0 and li_run >= 0:
        verdict = ('PASS（load_or_resume 行号在前 → before_train 晚于权重加载）'
                   if li_load < li_run else
                   'FAIL（顺序与预期相反！R0WeightCopyHook 须改惰性拷贝兜底）')
        print('  [机器判定·辅助] %s' % verdict)
    else:
        print('  [机器判定·辅助] 未能定位实际调用行，请按上方行号人工确认')

    print('\n[6.4] BaseModel.train_step 源码（parse_losses 调用证据）:')
    from mmengine.model import BaseModel
    print(inspect.getsource(BaseModel.train_step))

    print('[6.5] OptimWrapper 对 requires_grad=False 参数的处理（实测）:')
    from mmengine.optim import build_optim_wrapper
    net = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
    net[0].requires_grad_(False)
    ow = build_optim_wrapper(
        net, dict(type='OptimWrapper',
                  optimizer=dict(type='AdamW', lr=1e-2, weight_decay=0.5)))
    groups = ow.optimizer.param_groups
    print('  param_groups: %d 组，参数张量数=%s' %
          (len(groups), [len(g['params']) for g in groups]))
    frozen_before = net[0].weight.detach().clone()
    for _ in range(3):
        loss = net(torch.randn(8, 4)).pow(2).mean()
        ow.update_params(loss)
    unchanged = torch.equal(frozen_before, net[0].weight.detach())
    print('  3 步更新后冻结参数位级不变断言: %s' %
          ('PASS' if unchanged else 'FAIL —— 必须在 paramwise_cfg 中'
                                    '为冻结模块显式 lr_mult=0 兜底'))
    print('  （weight_decay=0.5 故意放大：若 FAIL 多半是 decay 作用于无梯度参数）')

    print('\n[6.6] paramwise_cfg custom_keys lr_mult 生效核验:')
    net2 = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
    ow2 = build_optim_wrapper(
        net2, dict(type='OptimWrapper',
                   optimizer=dict(type='AdamW', lr=2e-4),
                   paramwise_cfg=dict(custom_keys={
                       '0': dict(lr_mult=0.1)})))
    for g in ow2.optimizer.param_groups:
        print('  group lr=%g, 参数张量数=%d' % (g['lr'], len(g['params'])))
    print('  期望出现 lr=2e-05（lr_mult=0.1 命中 "0" 前缀的 Linear）与 lr=2e-4 两档')


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(
        description='EP-Fusion M0 运行时探针（只读；检测到训练进程即拒绝执行）')
    parser.add_argument('--config', default=DEFAULT_CONFIG,
                        help='复现 config 路径（默认 4xA30 config）')
    parser.add_argument('--repro-ckpt', default=None,
                        help='复现 checkpoint（默认建议 work_dirs/baseline1/epoch_19.pth；'
                             'epoch_20 为 CBGS 调度暴跌点，勿用）')
    parser.add_argument('--official-ckpt', default=None,
                        help='官方 checkpoint（...5239b1af.pth）')
    parser.add_argument('--work-dir', default=None,
                        help='复现训练 work_dir（解析日志用）')
    parser.add_argument('--out', default=None,
                        help='报告输出路径（默认 m0_probe_report_<日期>.txt）')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-sizes', default='1,2',
                        help='P4 显存矩阵 batch 列表（逗号分隔）')
    parser.add_argument('--timing-iters', type=int, default=3,
                        help='P4 每配置计时 iteration 数')
    parser.add_argument('--skip', default='',
                        help='跳过项，逗号分隔，如 p3,p4')
    args = parser.parse_args()

    out = args.out or ('m0_probe_report_%s.txt' %
                       datetime.datetime.now().strftime('%Y%m%d'))
    sys.stdout = Tee(out)

    print('EP-Fusion M0 探针报告')
    print('生成时间 : %s' % datetime.datetime.now().isoformat())
    print('VERSION  : %s（仓库根 VERSION 文件，铁律 16）' % read_version())
    print('仓库根   : %s' % REPO_ROOT)
    print('命令参数 : %s' % vars(args))

    # 铁律 13：训练进程自检（任何可疑即拒绝）
    suspects = detect_training_processes()
    if suspects:
        print('\n!!! 检测到疑似训练/GPU 进程，探针拒绝执行（铁律 13）:')
        for s in suspects:
            print('  ' + s)
        print('请等复现训练完全结束后再运行本探针。')
        sys.exit(2)
    print('\n[安全自检] 未检测到训练进程，继续。')

    skip = {s.strip().lower() for s in args.skip.split(',') if s.strip()}
    status = {}
    for name, fn in (('p1', probe_p1), ('p2', probe_p2), ('p3', probe_p3),
                     ('p4', probe_p4), ('p5', probe_p5), ('p6', probe_p6)):
        if name in skip:
            status[name] = 'SKIPPED(--skip)'
            continue
        try:
            fn(args)
            status[name] = 'DONE'
        except SystemExit:
            raise
        except Exception:
            status[name] = 'FAILED'
            print('\n[%s] 异常（不中断其余探针）:\n%s' %
                  (name.upper(), traceback.format_exc()))

    section('探针状态总览')
    for k, v in status.items():
        print('  %s: %s' % (k.upper(), v))
    print('\n报告已写入: %s（请整文件拷回）' % os.path.abspath(out))


if __name__ == '__main__':
    main()
