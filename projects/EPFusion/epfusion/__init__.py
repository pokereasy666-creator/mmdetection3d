# Copyright (c) EP-Fusion M0. All rights reserved.
"""EP-Fusion M0 包：显式 import 触发 @MODELS/@HOOKS 注册。

config 的 custom_imports=['projects.EPFusion.epfusion'] 会 import 本包，从而把
EPFusion / PoEFuser / EPFusionDataPreprocessor / EP-Fusion hooks 注册进 registry。
铁律 5：不导出 corruptions 的测试族（本模块内也无测试族实现）。
"""
from .corruptions import (TRAIN_CORRUPTIONS, apply_corruption,  # noqa: F401
                          make_generator)
from .ep_fusion import EPFusion  # noqa: F401
from .hooks import (PoETeacherInitHook, R0WeightCopyHook,  # noqa: F401
                    WTeachAnnealHook, copy_weights,
                    init_poe_from_teacher)
from .poe_fuser import PoEFuser  # noqa: F401
from .preprocessor import EPFusionDataPreprocessor  # noqa: F401

__all__ = [
    'EPFusion', 'PoEFuser', 'EPFusionDataPreprocessor', 'R0WeightCopyHook',
    'PoETeacherInitHook', 'WTeachAnnealHook', 'copy_weights',
    'init_poe_from_teacher', 'TRAIN_CORRUPTIONS', 'apply_corruption',
    'make_generator',
]
