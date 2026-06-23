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

## Final structure of DGFFuser (FAITHFUL — no camera suppression)
```
inputs = [img_bev (B,80,H,W), lidar_bev (B,256,H,W)]        # baseline ConvFuser order
 ├ lidar_proj : Conv2d(256→256,1x1)   = V_GB  (= W_Q)
 ├ img_proj   : Conv2d( 80→256,1x1)   = I_GB  (= W_K = W_V, shared)
 ├ P : 2D sine positional enc (256,H,W), param-free       # added to V_GB and I_GB (not value)
 ├ D : sin/cos of dist-to-centre matrix (256,H,W), param-free   # ⊙ onto the query
 ├ cross-attention (heads=8, head_dim=32) via F.scaled_dot_product_attention, scale 1/√32
 │     q=(V_GB+P)⊙D , k=I_GB+P , v=I_GB  → V̂_GB (B,256,H,W)        # FULL 180 (attn_resolution=None)
 ├ V̂_GB = out_proj(V̂_GB)              # 1x1 W_O, DEFAULT init (camera live from step 0)
 ├ U   = Norm1(V̂_GB + V_GB)            # added 1:1 — NO gamma gate
 └ F   = Norm2(FFN(U) + U)             # FFN = Conv3x3→ReLU→Conv3x3 ; out (B,256,H,W), signed
```
`Norm1`/`Norm2` `N` = **GroupNorm(32)** (feature-map norm, stats shared across space → **preserves
fg/bg contrast**). No final ReLU (output is norm-ended/signed). P and D are built lazily from the
actual (H,W), cached as **plain tensors** (not `nn.Parameter`, not buffers) so they never appear in
`parameters()` or `state_dict`.

> **No official DepthFusion code exists** (web search 2026-06 returned only the paper + unrelated
> repos). Round-2 history of `N`: channel-wise **LayerNorm was tried and REJECTED** by the 850-step
> smoke — per-cell LN normalises every BEV cell to norm √C, forcing fg/bg `contrast≡1.000` and
> freezing `loss_heatmap` (the heatmap head localises via spatial saliency). Eq.(4)'s "Add & Norm"
> in a **conv-on-feature-map** module (its FFN is 3×3 convs) is a **feature-map norm**, not per-token
> LN. → **GroupNorm**: shares stats across space (keeps contrast) and, like LN, has no batch stats /
> no cross-GPU sync (so it also avoids the SyncBN+fp16 nan). Separately, the `V̂` magnitude is bounded
> by **scaled-cosine attention** (A15), not by any camera gate. The camera is **never** suppressed.

## ASSUMPTIONs (paper-unspecified → chosen value + reason). Tagged in code.
| ID | Decision | Reason |
| --- | --- | --- |
| **A1** | `embed_dims = 256` | Paper uses C=128; we use 256 to match the baseline LiDAR-BEV / `pts_backbone(in_channels=256)` and **avoid an extra output projection**. |
| **A2** | Channel-align 1×1 convs ARE the attention projections: `W_Q=lidar_proj`, `W_K=W_V=img_proj` (key & value share the img projection). No separate per-head QKV linears. | Eq.(3) writes the attention directly on `(V_GB+P)⊙D`, `I_GB+P`, `I_GB`; the per-modality C-dim projection is the only learnable projection the paper shows. |
| **A2b** | Output projection `W_O` (`out_proj`, 1×1) is **always on, DEFAULT-initialised** (no zero-init). | Standard multi-head-attention output projection. Default init ⇒ the camera increment `V̂_GB` is **live from step 0** — the module is faithful and does **not** suppress the camera. (The earlier zero-init was a stability hack; removed.) |
| **A2c** | ~~ReZero scalar `gamma`~~ — **REMOVED**. Aggregation is `Norm1(V̂_GB + V_GB)` (added **1:1**). | Faithful Eq.(4) has no gate. LayerNorm symmetrically bounds `V̂+V_B`, so the camera cannot explode — the gate is unnecessary, and a gate is exactly what collapsed the camera (γ→0.014). No `gamma` param. |
| **A2d** | ~~Final `ReLU`~~ — **REMOVED**. Output is `Norm2(FFN(U)+U)` (LayerNorm-ended, **signed**). | Faithful Eq.(4) ends with `N` (LayerNorm), which is zero-mean/signed. `pts_backbone`'s first conv accepts signed input; matching ConvFuser's non-negativity was a cosmetic hack, not in the paper. |
| **A3** | `P` = parameter-free 2D sinusoidal PE (128 ch for y + 128 for x, temperature 1e4); added to query & key streams, **not** to value. | Paper says "positional encoding … added through element-wise addition" without details; value term in Eq.(3) is `I_GB` (no +P). |
| **A4** | `D` = sin/cos embedding (temperature 1e4) of the per-cell Euclidean distance to the centre cell `(H//2,W//2)`; **distance in BEV-cell-index units**. | Eq.(1)/(2) + "apply sine and cosine to the depth matrix"; units unspecified → cell units (resolution-agnostic). |
| **A5** | `num_heads = 8`, `head_dim = 32`. | Standard MHA; head_dim 32 satisfies flash-attention constraints. |
| **A6** | Attention scaling = **`1/√head_dim` = `1/√32`** (SDPA default). | In the multi-head realisation the **real per-head dim** `head_dim=32` is the correct scale (each head attends in a 32-dim subspace). Eq.(3) writes `1/√C` for the single-stream form; with C=256 split into 8 heads, `1/√32` is the self-consistent per-head equivalent. This is the intended "real head dim" scaling, not a deviation. |
| **A7** | SDPA forced to flash / mem-efficient backend, **math disabled**, on CUDA. | 180×180 = 32400-token global attention; math backend materialises ~32400² and OOMs 24GB. See "SDPA backend" below. |
| **A8** | `norm_cfg` default `None` → **GroupNorm** `dict(type='GN', num_groups=32)`: feature-map norm, stats shared across (group-channels × space). Configurable (BN2d via explicit cfg). | **Channel-wise LayerNorm was REJECTED** (850-step smoke): per-cell LN normalises each BEV cell to norm √C, forcing fg/bg `contrast_post≡1.000` (post-norm cell norms all = √256), `loss_heatmap` frozen 2.5–2.9, `matched_ious` 0.07. The heatmap localises via spatial magnitude → a per-cell-equalising norm destroys it. GN shares stats across space → **preserves contrast** (Stage-A `GN(v_gb)` preview: 2.4–7.2, all >1). Like LN, GN has no batch stats / no cross-GPU sync (avoids the SyncBN+fp16 nan; unaffected by `--sync_bn torch`). Honest: paper doesn't specify `N`; **GN is an adaptation** (a conv-FFN module ⇒ feature-map norm). Hard gate: measured full `contrast_post>1` (`[DGF-DEBUG contrast]`), else reject. |
| **A9** | FFN = `Conv3x3(256→256) → ReLU → Conv3x3(256→256)` (hidden ratio 1). Configurable via `ffn_channels`. | Paper: FFN "contains two convolution operations"; ratio/kernel unspecified → 3×3, ratio 1 (cheap, adds spatial mixing). |
| **A10** | Eq.(4) aggregation runs in **spatial (B,C,H,W)** layout; attention runs in token layout. | FFN = convolutions ⇒ spatial layout for Eq.(4). |
| **A11** | Attention dropout = 0. | Unspecified; default off. |
| **A12** | Residual base in Eq.(4) = `V_GB` (`lidar_proj`); output is lidar-centric, 256 ch. | Matches `V̂_GB + V_GB` in Eq.(4); 256 ch feeds `pts_backbone`. |
| **A13** | Common dim 256 instead of paper's 128. | Same as A1 (only `d_model` differs from the paper). |
| **A14** | `attn_resolution` default `None` = **FULL 180×180** attention (faithful, used by the +C config). The `R×R` downsample (`adaptive_avg_pool2d` → attention → `F.interpolate` up) is kept as a **dormant, off-by-default speed knob** for the later speed task only. | This task prioritises **faithful + stable**; speed is separate. Full res holds 32400 tokens via the forced flash/mem-efficient SDPA backend (math disabled). DGF_PERF showed the +C step is dominated by the attention **backward** ∝ N²; cutting `R` (e.g. 135: N 32400→18225, ~3.16× cheaper) is the lever for later — but it computes the camera increment at `R×R` then upsamples, a deviation from the paper, so it stays off here. No new params (pool/interp param-free). |

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
