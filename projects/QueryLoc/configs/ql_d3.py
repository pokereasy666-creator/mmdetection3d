# D3: 3 decoder layers, layers 2-3 copied from layer 1 at load time
# (decoder layer + prediction head incl. BN buffers), auxiliary supervision,
# evaluation on the last layer; multi-layer get_targets fixed (audit T1).
_base_ = ['./ql_protocol.py']
model = dict(
    bbox_head=dict(
        type='QueryLocTransFusionHead',
        num_decoder_layers=3,
        auxiliary=True,
        query_init='cell'))
