# Copyright (c) EP-Fusion M0. All rights reserved.
"""R0WeightCopyHook —— R0 对照组的 student_fuser 权重拷贝（M0_PLAN B.2 / F）。

R0（fusion_mode='convfuser'）用一个【可训练】的 ConvFuser 副本 student_fuser 替代 PoE，
其初值须从【冻结教师】fusion_layer 拷贝。拷贝必须发生在 checkpoint 加载之后：
P6 实证 Runner.train 中 load_or_resume(L63) 早于 train_loop.run(L75)，而 before_train 是
loop.run() 内首个 hook 点 ⇒ before_train 必然晚于 load_from 权重加载。

resume 守卫：仅在 runner.iter == 0（首次训练）执行拷贝；resume 续训时学生权重已训练，
严禁覆写。拷贝/跳过两条路径各打印一行带 run 信息的日志。
"""
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmdet3d.registry import HOOKS

__all__ = [
    'R0WeightCopyHook', 'PoETeacherInitHook', 'WTeachAnnealHook',
    'copy_weights', 'init_poe_from_teacher',
]


def copy_weights(model):
    """把 model.fusion_layer 权重逐张量拷给 model.student_fuser。

    抽成模块级函数：sanity_lambda_logging 不经 Runner.train()、before_train 不会自动触发，
    需在 load_checkpoint 后【显式调用】本函数再做权重相等断言（M0_PLAN G-1(e)）。

    Returns:
        bool: 成功拷贝返回 True；模型无 student_fuser/fusion_layer 时返回 False（no-op）。
    """
    if is_model_wrapper(model):
        model = model.module
    student = getattr(model, 'student_fuser', None)
    teacher = getattr(model, 'fusion_layer', None)
    if student is None or teacher is None:
        return False
    student.load_state_dict(teacher.state_dict())
    return True


def init_poe_from_teacher(model):
    """用冻结教师 ConvFuser 初始化 PoE 分支投影。"""
    if is_model_wrapper(model):
        model = model.module
    poe_fuser = getattr(model, 'poe_fuser', None)
    teacher = getattr(model, 'fusion_layer', None)
    if poe_fuser is None or teacher is None:
        return False
    return poe_fuser.init_from_convfuser(teacher)


@HOOKS.register_module()
class PoETeacherInitHook(Hook):
    """load_from 完成后的 before_train 时机初始化 PoE 分支投影。"""

    def before_train(self, runner):
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        if getattr(model, 'poe_fuser', None) is None:
            return
        version = _read_version()
        if runner.iter == 0:
            ok = init_poe_from_teacher(model)
            runner.logger.info(
                '[PoETeacherInitHook] initialized poe_fuser from fusion_layer '
                '@iter0 (ok=%s, epoch=%d, VERSION=%s)'
                % (ok, runner.epoch, version))
        else:
            runner.logger.info(
                '[PoETeacherInitHook] skip initialization '
                '(ok=SKIP, resume, iter=%d, epoch=%d, VERSION=%s)'
                % (runner.iter, runner.epoch, version))


@HOOKS.register_module()
class WTeachAnnealHook(Hook):
    """按 epoch 线性退火 w_teach；默认关闭。"""

    def __init__(self,
                 enable=False,
                 w_teach_end=0.0,
                 begin_epoch=0,
                 end_epoch=None,
                 mode='linear'):
        if mode != 'linear':
            raise ValueError('WTeachAnnealHook only supports mode="linear"')
        self.enable = bool(enable)
        self.w_teach_end = float(w_teach_end)
        self.begin_epoch = int(begin_epoch)
        self.end_epoch = None if end_epoch is None else int(end_epoch)
        self.mode = mode
        self.w_teach_start = None

    def before_train(self, runner):
        if not self.enable:
            runner.logger.info(
                '[WTeachAnnealHook] annealing disabled')
            return
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        self.w_teach_start = float(model.w_teach)
        if self.end_epoch is None:
            self.end_epoch = runner.max_epochs

    def before_train_epoch(self, runner):
        if not self.enable:
            return
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        span = self.end_epoch - self.begin_epoch
        if span <= 0:
            progress = 1.0 if runner.epoch >= self.end_epoch else 0.0
        else:
            progress = ((runner.epoch - self.begin_epoch) / float(span))
            progress = min(max(progress, 0.0), 1.0)
        model.w_teach = (
            self.w_teach_start
            + progress * (self.w_teach_end - self.w_teach_start))
        runner.logger.info(
            '[WTeachAnnealHook] epoch=%d w_teach=%.8g'
            % (runner.epoch, model.w_teach))


@HOOKS.register_module()
class R0WeightCopyHook(Hook):
    """before_train 时机把冻结教师 ConvFuser 权重拷给可训练 student_fuser（仅 R0 使用）。"""

    def before_train(self, runner):
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        # EP 主 config（poe 模式）无 student_fuser → no-op
        if getattr(model, 'student_fuser', None) is None:
            return
        version = _read_version()
        if runner.iter == 0:
            ok = copy_weights(model)
            runner.logger.info(
                '[R0WeightCopyHook] copied fusion_layer->student_fuser '
                '@iter0 (ok=%s, epoch=%d, VERSION=%s)'
                % (ok, runner.epoch, version))
        else:
            runner.logger.info(
                '[R0WeightCopyHook] skip copy (resume, iter=%d, epoch=%d, '
                'VERSION=%s)' % (runner.iter, runner.epoch, version))


def _read_version():
    """读仓库根 VERSION（铁律 16：不调 git；读不到记 unknown）。"""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, '..', '..', '..'))
    try:
        with open(os.path.join(root, 'VERSION'), encoding='utf-8') as f:
            return f.read().strip().splitlines()[0]
    except Exception:
        return 'unknown'
