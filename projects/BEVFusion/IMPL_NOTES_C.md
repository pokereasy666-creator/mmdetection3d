# IMPL_NOTES — Module C / DGFFuser (DepthFusion DGF)

> Module C = **Depth-GFusion (DGF)** from DepthFusion (arXiv:2505.07398) §III-B.
> Implemented file: `projects/BEVFusion/bevfusion/dgf_fuser.py`.
> **Only DGF** is implemented — NOT DLF (§III-C, overlaps with module D / InsFusion).
> Switched purely via config `model.fusion_layer.type` (`ConvFuser` → `DGFFuser`);
> baseline path is byte-identical when left as `ConvFuser`.

## Paper equations implemented (verified against the uploaded PDF)
- **Eq.(1)** `p_k = {(x_k,y_k) : d_k}, k∈[1,n]` — each BEV cell stores a depth value.
- **Eq.(2)** `d_k = E((x_k,y_k),(x_{n/2},y_{n/2}))` — Euclidean distance from each cell to the
  ego-centre cell; the depth matrix `M` → depth encoding `D` via sin/cos (no params).
- **Eq.(3)** `V̂_GB = softmax( ((V_GB+P)⊙D)(I_GB+P)ᵀ / √· ) · I_GB` — depth-modulated multi-head
  cross-attention. query=`(lidar+P)⊙D`, key=`img+P`, value=`img`.
- **Eq.(4)** `F_GB = N( FFN(N(V̂_GB+V_GB)) + N(V̂_GB+V_GB) )` — residual conv-FFN aggregation.

## Final structure of DGFFuser
```
inputs = [img_bev (B,80,H,W), lidar_bev (B,256,H,W)]        # baseline ConvFuser order
 ├ lidar_proj : Conv2d(256→256,1x1)   = V_GB  (= W_Q)
 ├ img_proj   : Conv2d( 80→256,1x1)   = I_GB  (= W_K = W_V, shared)
 ├ P : 2D sine positional enc (256,H,W), param-free       # added to V_GB and I_GB (not value)
 ├ D : sin/cos of dist-to-centre matrix (256,H,W), param-free   # ⊙ onto the query
 ├ cross-attention (heads=8, head_dim=32) via F.scaled_dot_product_attention
 │     q=(V_GB+P)⊙D , k=I_GB+P , v=I_GB  → V̂_GB (B,256,H,W)
 ├ V̂_GB = out_proj(V̂_GB)              # 1x1, ZERO-init -> 0 at step 0  [fix-residual-stability]
 ├ x   = Norm1(V_GB + gamma * V̂_GB)    # gamma=ReZero scalar init 0.05  [fix-residual-stability]
 ├ out = Norm2(FFN(x) + x)             # FFN = Conv3x3→ReLU→Conv3x3
 └ out = ReLU(out)                     # non-negative, matches ConvFuser ; out (B,256,H,W)  [fix-residual-stability]
```
P and D are built lazily from the actual (H,W), cached as **plain tensors** (not
`nn.Parameter`, not buffers) so they never appear in `parameters()` or `state_dict`.

## ASSUMPTIONs (paper-unspecified → chosen value + reason). Tagged in code.
| ID | Decision | Reason |
| --- | --- | --- |
| **A1** | `embed_dims = 256` | Paper uses C=128; we use 256 to match the baseline LiDAR-BEV / `pts_backbone(in_channels=256)` and **avoid an extra output projection**. |
| **A2** | Channel-align 1×1 convs ARE the attention projections: `W_Q=lidar_proj`, `W_K=W_V=img_proj` (key & value share the img projection). No separate per-head QKV linears. | Eq.(3) writes the attention directly on `(V_GB+P)⊙D`, `I_GB+P`, `I_GB`; the per-modality C-dim projection is the only learnable projection the paper shows. |
| **A2b** | Output projection `W_O` (`out_proj`, 1×1) is **always on and ZERO-initialised** (weight & bias = 0). | [module-C/fix-residual-stability] Makes the camera increment `V̂_GB = out_proj(attn) = 0` at step 0 → fusion starts as a pure-LiDAR(-derived) path, killing the camera-increment explosion (|v_hat| was up to ~390× |v_gb|). `out_proj` still gets gradient (∝ gamma≠0) and learns out of zero. (Replaces the earlier optional `use_out_proj`; arg removed.) |
| **A2c** | **ReZero scalar `gamma`** on the camera increment: `Norm1(V_GB + gamma·V̂)`, `gamma` init **0.05 (nonzero)**. | [module-C/fix-residual-stability] Gates camera-increment magnitude. **NONZERO on purpose**: `gamma=0` *with* zero-init `out_proj` would freeze the camera path (∂L/∂gamma ∝ v_hat=0 and ∂L/∂out_proj ∝ gamma=0). 0.05 keeps the increment small while letting both learn. New param in state_dict (only under +C). |
| **A2d** | **Final `ReLU`** after `Norm2`. | [module-C/fix-residual-stability] Matches ConvFuser's Conv-BN-**ReLU** non-negative output; removes the extreme negative values (e.g. −92) DGF produced that mismatched `pts_backbone`'s expected distribution. |
| **A3** | `P` = parameter-free 2D sinusoidal PE (128 ch for y + 128 for x, temperature 1e4); added to query & key streams, **not** to value. | Paper says "positional encoding … added through element-wise addition" without details; value term in Eq.(3) is `I_GB` (no +P). |
| **A4** | `D` = sin/cos embedding (temperature 1e4) of the per-cell Euclidean distance to the centre cell `(H//2,W//2)`; **distance in BEV-cell-index units**. | Eq.(1)/(2) + "apply sine and cosine to the depth matrix"; units unspecified → cell units (resolution-agnostic). |
| **A5** | `num_heads = 8`, `head_dim = 32`. | Standard MHA; head_dim 32 satisfies flash-attention constraints. |
| **A6** | Attention scaling = **`1/√head_dim`** (SDPA default), **NOT** the paper's `1/√C`. | `F.scaled_dot_product_attention` applies `1/√head_dim`; per-head scaling is the conventional/correct choice. **This is an explicit deviation from Eq.(3)'s `1/√C`.** If exact paper parity is wanted, pass `scale=1/sqrt(embed_dims)` to SDPA. |
| **A7** | SDPA forced to flash / mem-efficient backend, **math disabled**, on CUDA. | 180×180 = 32400-token global attention; math backend materialises ~32400² and OOMs 24GB. See "SDPA backend" below. |
| **A8** | `norm_cfg` default `dict(type='GN', num_groups=32)` → **GroupNorm** (32 groups, 8 ch/group). Configurable. | **NOT BatchNorm2d** [module-C/bn-to-groupnorm]. `norm1` runs on the sparse, low-variance `lidar_proj(lidar_bev)` output (norm~229, std~0.056, lots of zeros); BN's per-channel `(x-μ)/√(σ²+eps)` renormalises std~0.056 to ~1, amplifying the feature norm ~18× (229→~4218) → `grad_norm=NaN`, `loss_heatmap` explodes (~14 vs ~4). Diagnosed via the DGF_DEBUG step-norms (Round 8). GroupNorm is batch-independent (no running stats; also unaffected by `--sync_bn torch`) and does not blow up sparse features; attention/transformer-style blocks normally use GN/LN, not BN. Override to `dict(type='BN2d')` only to restore BN. |
| **A9** | FFN = `Conv3x3(256→256) → ReLU → Conv3x3(256→256)` (hidden ratio 1). Configurable via `ffn_channels`. | Paper: FFN "contains two convolution operations"; ratio/kernel unspecified → 3×3, ratio 1 (cheap, adds spatial mixing). |
| **A10** | Eq.(4) aggregation runs in **spatial (B,C,H,W)** layout; attention runs in token layout. | FFN = convolutions ⇒ spatial layout for Eq.(4). |
| **A11** | Attention dropout = 0. | Unspecified; default off. |
| **A12** | Residual base in Eq.(4) = `V_GB` (`lidar_proj`); output is lidar-centric, 256 ch. | Matches `V̂_GB + V_GB` in Eq.(4); 256 ch feeds `pts_backbone`. |
| **A13** | Common dim 256 instead of paper's 128. | Same as A1 (only `d_model` differs from the paper). |

## SDPA backend — how to CONFIRM flash/mem-efficient is used (A7)
The forward wraps the attention in `efficient_sdpa_ctx()` **only on CUDA**, which enables FLASH +
EFFICIENT and **disables MATH**. Because MATH is disabled, if no efficient kernel is eligible the
call **raises** (e.g. `RuntimeError: No available kernel`) instead of silently OOM-ing — that is
itself the guard. To positively confirm the backend at runtime:

- torch 2.0.x (the pinned env): the `torch.backends.cuda.sdp_kernel(enable_math=False, …)` context
  is active during the call; a successful forward ⇒ flash or mem-efficient ran (math was off).
- Optional probe (any torch): run a tiny `(1,8,32400,32)` fp16 cuda SDPA inside the context and
  time / profile it; or use `torch.backends.cuda.flash_sdp_enabled()` /
  `mem_efficient_sdp_enabled()` to see which are enabled.
- torch ≥ 2.1: replace with `torch.nn.attention.sdpa_kernel([FLASH_ATTENTION, EFFICIENT_ATTENTION])`
  (the code already prefers this API when available); you can also wrap the call in
  `torch.profiler` and check the kernel name contains `flash`/`efficient`.

Conditions for flash: fp16/bf16 (true under `--amp`), head_dim 32 (OK), Ampere+ (A30 = sm_80, OK).
Without `--amp` (fp32) flash is unavailable → the **mem-efficient** backend handles fp32 (still
O(N) memory). On CPU (unit tests) the context is skipped and the math backend runs (256 tokens).

## Files (this task)
- `projects/BEVFusion/bevfusion/dgf_fuser.py` — DGFFuser (+ helpers).
- `projects/BEVFusion/bevfusion/__init__.py` — `+from .dgf_fuser import DGFFuser` (additive only).
- `projects/BEVFusion/configs/bevfusion_lidar-cam_4xa30_dgf_nus-3d.py` — +C config.
- `projects/BEVFusion/tests/test_dgf_fuser.py` — CPU unit tests.
- `tools/check_c_alloff.py` — baseline-untouched check.
- `projects/BEVFusion/RUNBOOK_C.md` — offline run/smoke/OOM guide (results PENDING).

## Out of scope (intentionally NOT done)
- DLF (DepthFusion §III-C) — overlaps with module D (InsFusion).
- Any change to module A / D, or to baseline code beyond the one additive import.
- Any number (memory / mAP / NDS) — all PENDING, to be measured on the 4×A30 server.
