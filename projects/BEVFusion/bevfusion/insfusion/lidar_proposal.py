import torch
from mmcv.cnn import ConvModule, build_conv_layer
from torch import Tensor, nn

from mmdet3d.registry import MODELS

from ..proposal_utils import build_proposal_pack, create_2d_grid
from ..structures import ProposalPack


@MODELS.register_module()
class LiDARProposalGenerator(nn.Module):
    """Generate heatmap proposals from the pre-fusion LiDAR BEV tensor.

    The returned dense heatmap is the boundary for a future supervised
    proposal loss. Milestone 1 intentionally does not implement that loss.
    """

    def __init__(self,
                 in_channels: int = 256,
                 hidden_channel: int = 128,
                 num_classes: int = 10,
                 num_proposals: int = 300,
                 nms_kernel_size: int = 3,
                 dataset: str = 'nuScenes',
                 bn_momentum: float = 0.1) -> None:
        super().__init__()
        self.num_proposals = num_proposals
        self.nms_kernel_size = nms_kernel_size
        self.dataset = dataset

        self.shared_conv = build_conv_layer(
            dict(type='Conv2d'),
            in_channels,
            hidden_channel,
            kernel_size=3,
            padding=1,
        )
        self.heatmap_head = nn.Sequential(
            ConvModule(
                hidden_channel,
                hidden_channel,
                kernel_size=3,
                padding=1,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
            ),
            build_conv_layer(
                dict(type='Conv2d'),
                hidden_channel,
                num_classes,
                kernel_size=3,
                padding=1,
            ),
        )
        self.class_encoding = nn.Conv1d(num_classes, hidden_channel, 1)
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.momentum = bn_momentum

    def forward(self, lidar_bev: Tensor) -> ProposalPack:
        feature = self.shared_conv(lidar_bev)
        with torch.autocast('cuda', enabled=False):
            dense_heatmap = self.heatmap_head(feature.float())
        bev_pos = create_2d_grid(
            feature.shape[-1], feature.shape[-2], dtype=torch.float32)
        return build_proposal_pack(
            feature=feature,
            dense_heatmap=dense_heatmap,
            bev_pos=bev_pos,
            class_encoding=self.class_encoding,
            num_proposals=self.num_proposals,
            nms_kernel_size=self.nms_kernel_size,
            dataset=self.dataset,
            source='lidar',
        )
