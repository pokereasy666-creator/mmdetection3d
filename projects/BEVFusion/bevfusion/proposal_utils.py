from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .structures import ProposalPack


def create_2d_grid(width: int, height: int, device=None, dtype=None) -> Tensor:
    """Create BEV cell-center coordinates with shape ``[1, H*W, 2]``."""
    x = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    y = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    batch_x, batch_y = torch.meshgrid(x, y)
    coordinates = torch.cat([(batch_x + 0.5)[None], (batch_y + 0.5)[None]],
                            dim=0)
    return coordinates[None].view(1, 2, -1).permute(0, 2, 1)


def _special_local_max_classes(dataset: str) -> Sequence[int]:
    if dataset == 'nuScenes':
        return (8, 9)
    if dataset == 'Waymo':
        return (1, 2)
    return ()


def local_maximum_heatmap(heatmap: Tensor, kernel_size: int,
                          dataset: str) -> Tensor:
    """Apply the local-maximum filtering used by TransFusion proposals."""
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError('kernel_size must be a positive odd integer')

    if kernel_size == 1:
        local_max = heatmap
    else:
        padding = kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        local_max_inner = F.max_pool2d(
            heatmap, kernel_size=kernel_size, stride=1, padding=0)
        local_max[:, :, padding:-padding, padding:-padding] = local_max_inner

    for class_index in _special_local_max_classes(dataset):
        if class_index < heatmap.shape[1]:
            local_max[:, class_index] = heatmap[:, class_index]
    return heatmap * (heatmap == local_max)


def build_proposal_pack(feature: Tensor, dense_heatmap: Tensor,
                        bev_pos: Tensor, class_encoding: nn.Module,
                        num_proposals: int, nms_kernel_size: int, dataset: str,
                        source: str) -> ProposalPack:
    """Select heatmap peaks and gather their feature and position queries."""
    batch_size = feature.shape[0]
    feature_flatten = feature.view(batch_size, feature.shape[1], -1)
    heatmap = dense_heatmap.detach().sigmoid()
    heatmap = local_maximum_heatmap(heatmap, nms_kernel_size, dataset)
    heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)

    if num_proposals > heatmap.shape[1] * heatmap.shape[2]:
        raise ValueError('num_proposals exceeds the available heatmap cells')

    top_proposals = heatmap.view(batch_size, -1).argsort(
        dim=-1, descending=True)[..., :num_proposals]
    top_classes = top_proposals // heatmap.shape[-1]
    top_indices = top_proposals % heatmap.shape[-1]

    query_feat_pre = feature_flatten.gather(
        index=top_indices[:, None, :].expand(-1, feature.shape[1], -1),
        dim=-1,
    )
    one_hot = F.one_hot(
        top_classes, num_classes=heatmap.shape[1]).permute(0, 2, 1)
    query_feat_post = query_feat_pre + class_encoding(one_hot.float())

    if bev_pos.shape[0] == 1:
        batch_bev_pos = bev_pos.repeat(batch_size, 1, 1).to(feature.device)
    elif bev_pos.shape[0] == batch_size:
        batch_bev_pos = bev_pos.to(feature.device)
    else:
        raise ValueError('bev_pos batch dimension must be one or batch_size')
    query_pos = batch_bev_pos.gather(
        index=top_indices[:,
                          None, :].permute(0, 2,
                                           1).expand(-1, -1,
                                                     batch_bev_pos.shape[-1]),
        dim=1,
    )
    class_scores = heatmap.gather(
        index=top_indices[:, None, :].expand(-1, heatmap.shape[1], -1),
        dim=-1,
    )
    scores = heatmap.view(batch_size, -1).gather(1, top_proposals)
    return ProposalPack(
        query_feat_pre=query_feat_pre.transpose(1, 2),
        query_feat_post=query_feat_post.transpose(1, 2),
        ref_xy=query_pos,
        scores=scores,
        class_scores=class_scores.transpose(1, 2),
        labels=top_classes,
        indices=top_indices,
        dense_heatmap=dense_heatmap,
        source_type=source,
    )
