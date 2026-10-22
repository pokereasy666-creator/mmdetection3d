# M-a: unbiased heatmap target + DARK initial positions (sec. 7.2), no new
# parameters. Base with 1 decoder layer (see ql_mb.py for 3 layers).
_base_ = ['./ql_protocol.py']
model = dict(
    bbox_head=dict(
        type='QueryLocTransFusionHead',
        query_init='dark',
        feat_sampling='bilinear',
        dark_clamp=0.5,
        unbiased_heatmap=True))
