# Copyright (c) EP-Fusion M0. All rights reserved.
"""EP-Fusion M0 包：显式 import 触发 @MODELS/@HOOKS 注册。

config 的 custom_imports=['projects.EPFusion.epfusion'] 会 import 本包，从而把
EPFusion / PoEFuser / EPFusionDataPreprocessor / R0WeightCopyHook 注册进 registry。
铁律 5：不导出 corruptions 的测试族（本模块内也无测试族实现）。
"""
from .corruptions import (TRAIN_CORRUPTIONS, apply_corruption,  # noqa: F401
                          make_generator)
from .ep_fusion import EPFusion  # noqa: F401
from .hooks import R0WeightCopyHook, copy_weights  # noqa: F401
from .poe_fuser import PoEFuser  # noqa: F401
from .preprocessor import EPFusionDataPreprocessor  # noqa: F401

__all__ = [
    'EPFusion', 'PoEFuser', 'EPFusionDataPreprocessor', 'R0WeightCopyHook',
    'copy_weights', 'TRAIN_CORRUPTIONS', 'apply_corruption', 'make_generator',
]
