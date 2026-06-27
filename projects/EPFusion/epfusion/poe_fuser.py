# Copyright (c) EP-Fusion M0. All rights reserved.
"""PoEFuser —— 替换 ConvFuser 的逐格标量 Λ 精度加权 PoE 融合（M0_PLAN B.1）。

设计要点（铁律 6 / I-10）：
- proj_C/proj_L：1x1 conv 把相机/LiDAR BEV 投影到共享维 D（探针 P3 坐实 80/256→256）；
- lambda_head_C/L：2 个 3x3 conv + 1 个 1x1 conv 输出逐格 log-precision，clamp[-7,7]，
  末层零初始化 ⇒ 初始 logΛ≡0 ⇒ Λ≡1；
- PoE 闭式融合 μ_F = (Λ_C·P_C + Λ_L·P_L) / (Λ_C + Λ_L + eps)；
- 全程 fp32（局部关 autocast + 显式 .float()），避免 fp16 下 Λ·diff² 溢出。
"""
import torch
import torch.nn as nn

__all__ = ['PoEFuser']


class PoEFuser(nn.Module):
    """逐格标量 Λ 的 product-of-experts 融合层。

    Args:
        in_channels_cam (int): 相机 BEV 通道（默认 80，P3 坐实）。
        in_channels_lidar (int): LiDAR BEV 通道（默认 256，P3 坐实）。
        embed_dims (int): 共享投影维 D（默认 256 = pts_backbone 入口）。
        lambda_hidden (int): Λ 头隐层宽（默认 64）。
        gn_groups (int): GroupNorm 组数（默认 8）。
        clamp_min/clamp_max (float): log-precision clamp 区间（默认 [-7, 7]）。
        eps (float): PoE 分母数值保护（默认 1e-6）。
        lambda_input (str): 'raw'（投影前原始分支特征，默认）或 'projected'。
        norm (str): 'GN'（默认）或 'none'。
    """

    def __init__(self,
                 in_channels_cam=80,
                 in_channels_lidar=256,
                 embed_dims=256,
                 lambda_hidden=64,
                 gn_groups=8,
                 clamp_min=-7.0,
                 clamp_max=7.0,
                 eps=1e-6,
                 lambda_input='raw',
                 norm='GN'):
        super().__init__()
        assert lambda_input in ('raw', 'projected')
        assert norm in ('GN', 'none')
        self.embed_dims = embed_dims
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.eps = float(eps)
        self.lambda_input = lambda_input

        self.proj_C = nn.Conv2d(in_channels_cam, embed_dims, kernel_size=1)
        self.proj_L = nn.Conv2d(in_channels_lidar, embed_dims, kernel_size=1)

        lam_in_C = in_channels_cam if lambda_input == 'raw' else embed_dims
        lam_in_L = in_channels_lidar if lambda_input == 'raw' else embed_dims
        self.lambda_head_C = self._build_lambda_head(lam_in_C, lambda_hidden,
                                                     gn_groups, norm)
        self.lambda_head_L = self._build_lambda_head(lam_in_L, lambda_hidden,
                                                     gn_groups, norm)
        # 末层零初始化 ⇒ Λ≡1（铁律 6）
        self._zero_init_last(self.lambda_head_C)
        self._zero_init_last(self.lambda_head_L)

    @staticmethod
    def _build_lambda_head(c_in, hidden, gn_groups, norm):
        layers = [nn.Conv2d(c_in, hidden, kernel_size=3, padding=1)]
        if norm == 'GN':
            layers.append(nn.GroupNorm(gn_groups, hidden))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(hidden, hidden, kernel_size=3, padding=1))
        if norm == 'GN':
            layers.append(nn.GroupNorm(gn_groups, hidden))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(hidden, 1, kernel_size=1))
        return nn.Sequential(*layers)

    @staticmethod
    def _zero_init_last(head):
        last = head[-1]
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)

    def forward(self, F_C, F_L):
        """Args: F_C [B,Ccam,H,W], F_L [B,Clidar,H,W]（来自冻结分支，无梯度）。

        Returns: dict(mu_F, log_lambda_C, log_lambda_L, lam_C, lam_L, pc, pl)。
        """
        # I-10：PoE 与下游 NLL 一律 fp32，关 autocast 防 Λ·diff² 溢出
        with torch.autocast(device_type='cuda', enabled=False):
            F_C = F_C.float()
            F_L = F_L.float()
            pc = self.proj_C(F_C)
            pl = self.proj_L(F_L)
            in_C = F_C if self.lambda_input == 'raw' else pc
            in_L = F_L if self.lambda_input == 'raw' else pl
            log_C = torch.clamp(self.lambda_head_C(in_C),
                                self.clamp_min, self.clamp_max)
            log_L = torch.clamp(self.lambda_head_L(in_L),
                                self.clamp_min, self.clamp_max)
            lam_C = torch.exp(log_C)
            lam_L = torch.exp(log_L)
            mu = (lam_C * pc + lam_L * pl) / (lam_C + lam_L + self.eps)
        return dict(mu_F=mu, log_lambda_C=log_C, log_lambda_L=log_L,
                    lam_C=lam_C, lam_L=lam_L, pc=pc, pl=pl)
