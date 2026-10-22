# M-b: peak offset initialization (sec. 7.1). Offset head on the shared BEV
# feature, last layer zero-initialised, tanh x 3 cells, delta detached,
# dense L1 (radius 3, lambda 0.25), lr_mult 10 for the offset head only.
# Base with 1 decoder layer; if Q5 selects 3 layers, add
#   model.bbox_head.num_decoder_layers=3
_base_ = ['./ql_protocol.py']
model = dict(
    bbox_head=dict(
        type='QueryLocTransFusionHead',
        query_init='offset',
        feat_sampling='bilinear',
        offset_channels=128,
        offset_range=3.0,
        offset_sup_radius=3,
        loss_offset_weight=0.25))
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={'bbox_head.offset_head': dict(lr_mult=10)}))
