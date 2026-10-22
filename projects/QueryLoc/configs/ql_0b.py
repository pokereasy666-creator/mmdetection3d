# Step 0b probe (test only, start weights): initial query_pos = DARK decode,
# query features still gathered at the peak cell.
_base_ = ['./ql_protocol.py']
model = dict(
    bbox_head=dict(
        type='QueryLocTransFusionHead',
        query_init='dark',
        feat_sampling='cell',
        dark_clamp=0.5))
