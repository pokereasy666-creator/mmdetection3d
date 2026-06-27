# Copyright (c) EP-Fusion M0. All rights reserved.
"""EPFusionDataPreprocessor —— 模式采样 + 单模态退化注入 + 干净副本携带（M0_PLAN D.2）。

机制（候选 2）：子类化 Det3DDataPreprocessor，在【归一化前】注入损坏（铁律 3）：
- 图像损坏作用于 collate_data 归一化之前的 0-255 float 多视角图；
- 点云损坏作用于体素化（模型内）之前的原始点（preprocessor 不体素化、points 透传）。
逐 micro-batch 整批同一模式采样 clean/corrupt_cam/corrupt_lidar；损坏模式内以 p_zero 概率抽
整路置零条目，否则均匀抽一种普通损坏 + s~U(severity_range)。模式经 inputs['corrupt_mode'] 传入模型。
干净副本（emit_clean=True 时）经同一 super().simple_process 归一化后挂 inputs['imgs_clean']/['points_clean']
供教师前向；R0（emit_clean=False）不生成副本、省第二次归一化。
"""
import torch

from mmdet3d.models.data_preprocessors.data_preprocessor import \
    Det3DDataPreprocessor
from mmdet3d.registry import MODELS

from .corruptions import (NORMAL_ENTRIES, ZERO_ENTRIES, apply_corruption,
                          make_generator)

__all__ = ['EPFusionDataPreprocessor']


@MODELS.register_module()
class EPFusionDataPreprocessor(Det3DDataPreprocessor):
    """在归一化前注入单模态退化、并按需携带干净副本的预处理器。

    Args:
        mode_probs (tuple): (clean, corrupt_cam, corrupt_lidar) 概率，默认 (0.5,0.25,0.25)。
        p_zero (float): 损坏模式内抽中整路置零条目的概率，默认 0.05。
        severity_range (tuple): 普通损坏严重度采样区间，默认 (0.1, 1.0)。
        seed (int): rng 基种子（与 randomness.seed 对齐），默认 2026。
        emit_clean (bool): 是否生成/携带干净副本（R1-R3 True；R0 False）。
        force_mode/force_corruption/force_severity: 诊断脚本强制用（默认 None）。
    """

    def __init__(self,
                 *args,
                 mode_probs=(0.5, 0.25, 0.25),
                 p_zero=0.05,
                 severity_range=(0.1, 1.0),
                 seed=2026,
                 emit_clean=True,
                 force_mode=None,
                 force_corruption=None,
                 force_severity=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.mode_probs = tuple(mode_probs)
        self.p_zero = float(p_zero)
        self.severity_range = tuple(severity_range)
        self.seed = int(seed)
        self.emit_clean = bool(emit_clean)
        self.force_mode = force_mode
        self.force_corruption = force_corruption
        self.force_severity = force_severity
        self._iter = 0

    # ---------------------------------------------------------------- 采样
    def _rng(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        g = make_generator(self.seed, rank, self._iter)
        self._iter += 1
        return g

    def _sample_mode(self, g):
        r = torch.rand(1, generator=g).item()
        c, cc, _cl = self.mode_probs
        if r < c:
            return 'clean'
        if r < c + cc:
            return 'corrupt_cam'
        return 'corrupt_lidar'

    def _pick(self, modality, g):
        if self.force_corruption is not None:
            name = self.force_corruption
        elif torch.rand(1, generator=g).item() < self.p_zero:
            name = ZERO_ENTRIES[modality]
        else:
            cand = NORMAL_ENTRIES[modality]
            idx = int(torch.randint(0, len(cand), (1,), generator=g).item())
            name = cand[idx]
        if self.force_severity is not None:
            sev = float(self.force_severity)
        else:
            lo, hi = self.severity_range
            sev = float(lo + (hi - lo) * torch.rand(1, generator=g).item())
        return name, sev

    # ---------------------------------------------------------------- 主流程
    def simple_process(self, data, training=False):
        mode = self.force_mode
        if mode is None:
            if not training:
                return super().simple_process(data, training)
            g = self._rng()
            mode = self._sample_mode(g)
        else:
            g = self._rng()

        if mode == 'clean':
            out = super().simple_process(data, training)
            out['inputs']['corrupt_mode'] = 'clean'
            out['inputs']['corrupt_name'] = 'none'
            out['inputs']['corrupt_severity'] = 0.0
            return out

        inputs = data['inputs']
        name, sev = self._pick('img' if mode == 'corrupt_cam' else 'pts', g)

        if mode == 'corrupt_cam':
            imgs = inputs.get('img', None)
            clean = ([t.clone() for t in imgs]
                     if (self.emit_clean and imgs is not None) else None)
            if imgs is not None:
                inputs['img'] = [apply_corruption('img', name, t, sev, g)
                                 for t in imgs]
            out = super().simple_process(data, training)
            if clean is not None:
                # data_samples=None：仅归一化 clean 图，不重复处理 gt
                clean_out = super().simple_process(
                    {'inputs': {'img': clean}, 'data_samples': None}, training)
                out['inputs']['imgs_clean'] = clean_out['inputs']['imgs']
        else:  # corrupt_lidar
            points = inputs.get('points', None)
            clean = ([t.clone() for t in points]
                     if (self.emit_clean and points is not None) else None)
            if points is not None:
                inputs['points'] = [apply_corruption('pts', name, t, sev, g)
                                    for t in points]
            out = super().simple_process(data, training)
            if clean is not None:
                dev = out['inputs']['points'][0].device
                out['inputs']['points_clean'] = [t.to(dev) for t in clean]

        out['inputs']['corrupt_mode'] = mode
        out['inputs']['corrupt_name'] = name
        out['inputs']['corrupt_severity'] = sev
        return out
