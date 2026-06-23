# RUNBOOK — Module C / +C (DGFFuser) on the offline 4×A30 server

> Deploy = download the `ablation/dgf` branch zip → upload → unzip (overwrites
> source tree). **No git, no network on the server.** All result numbers below
> are `PENDING` — run on the server and fill in; **do not fabricate.**

## 0. After every re-extract: rebuild the CUDA ops
The `.so` are git-ignored and live in the source tree, so they vanish on each
re-unzip. Rebuild (needs the env's nvcc, see the earlier env runbook):
```bash
cd <repo-root>
export CUDA_HOME=$CONDA_PREFIX        # if nvcc came from the conda env
FORCE_CUDA=1 python projects/BEVFusion/setup.py develop
python -c "from projects.BEVFusion.bevfusion.ops import bev_pool, Voxelization; print('ops OK')"
```

## 1. Placeholders to replace (offline, local absolute paths only)
- In `projects/BEVFusion/configs/bevfusion_lidar-cam_4xa30_dgf_nus-3d.py`:
  - `work_dir = '/data/abl/dgf'` → **a real absolute path OUTSIDE the repo**
    (so logs/checkpoints survive re-extraction). Or pass `--work-dir` on the CLI.
- At launch via `--cfg-options` (LOCAL paths, never URLs):
  - `load_from=<LIDAR_CKPT_PATH>` → local `…/bevfusion_lidar_voxel0075…-2628f933.pth`
  - `model.img_backbone.init_cfg.checkpoint=<SWINT_CKPT_PATH>` → local `…/swint-nuimages-pretrained.pth`

## 2. CPU unit test (no GPU/ops needed) — sanity before training
```bash
cd <repo-root>
pytest projects/BEVFusion/tests/test_dgf_fuser.py -q
```
Checks output shape, that D/P carry no trainable params, backward runs,
resolution-agnostic, the faithful contract (no `gamma`, no `out_relu`, `out_proj`
NOT zero-init, signed output), that the camera contributes, that the default norm
is **GroupNorm(32)** and that GroupNorm **preserves fg/bg contrast** (LN's
rejection guard). (Needs `pytest`; if unavailable, run the module directly.)

## 3. Baseline-untouched check (needs compiled ops; CPU is fine)
```bash
cd <repo-root>
python tools/check_c_alloff.py
```
Asserts baseline=ConvFuser, +C=DGFFuser, and that **only** `fusion_layer.*`
keys change (every other parameter identical) ⇒ module C touches nothing else.

## 4. Smoke test BEFORE the full run — FULL 180, with DGF diagnostics ON
Run ~200–300 iters (don't run a full epoch — 180 is slow) with `DGF_DEBUG=1` and
watch the `[DGF-DEBUG]` lines:
```bash
LIDAR_CKPT=<LIDAR_CKPT_PATH>
SWINT_CKPT=<SWINT_CKPT_PATH>
CFG=projects/BEVFusion/configs/bevfusion_lidar-cam_4xa30_dgf_nus-3d.py

DGF_DEBUG=1 DGF_DEBUG_EVERY=50 \
bash tools/dist_train.sh ${CFG} 4 \
  --amp --sync_bn torch \
  --work-dir /data/abl/dgf_smoke \
  --cfg-options \
    train_cfg.max_epochs=1 \
    load_from=${LIDAR_CKPT} \
    model.img_backbone.init_cfg.checkpoint=${SWINT_CKPT}
```
Confirm in the log (the new `fusion_layer.*` DGF weights are expected in
`missing_keys` w.r.t. the LiDAR-only ckpt — normal, they train from scratch).
**Quantitative pass/fail gates** (RED = stop and diagnose, do NOT add suppression):
- **No NaN/Inf** fwd+bwd: the `[DGF-NAN]` localizer must stay **silent**. If it
  fires, it names the FIRST bad tensor (`q/k/v_hat/u/ffn_out/out`) or grad —
  fix the cause. If overflow only (inf), drop `loss_scale 64 → 32` via
  `--cfg-options optim_wrapper.loss_scale=32.0` and re-smoke. (64 is a starting
  point; if healthy, keep it.)
- **BUG-2 / attention magnitude** (`[DGF-DEBUG bug2]` + `[DGF-DEBUG]`): `raw_ratio`
  `|vhat|/|vgb|` must be **O(1) (~0.3–3) and STABLE** across the entropy swing — NOT
  the old erratic 7–139×. RED if `max_prob → ~1` / `attn_entropy → ~0` (one-hot), or
  if per-head `qk_scale g` is **pinned at `max_allowed`** (`pinned=8/8`) with entropy
  sliding to ~7.7 (ceiling too low / broadcast persists). `|attn_out| ≈ |vhat|`
  should now track `|i_gb|`, not explode.
- **BUG-1 / contrast** (`[DGF-DEBUG contrast]` + `[DGF-DEBUG bug1-isolated]`): the GATE
  is `contrast_post(fg/bg) > 1` (JOINT, from GN norm1). The isolated
  `contrast_gn(v_gb)` reads GN ALONE (>1 whenever GN works) — use it to attribute if
  the joint number lags while BUG-2 settles. If `contrast_post ≈ 1` once BUG-2 is
  stable → GN failed, reject (IMPL_NOTES_C A8).
- **Camera contributes**: `rel_cam(||F-Flidar||/||Flidar||)` **stably > 0.05**. RED at ~0.
- **No camera gating**: structurally guaranteed (no `gamma`; QK-norm is modality-neutral).
- **Memory fits** (no CUDA OOM at full 180) — note the peak (step 6). QK-norm keeps the
  SDPA flash/mem-efficient path (O(N)), so memory is unchanged vs the pre-fix run.
- **loss finite and trending down**; send the loss / grad_norm curves + the gate
  table for sign-off BEFORE the long run.

## 5. Full +C training (same recipe as baseline; fairness rule)
Drop `train_cfg.max_epochs=1` and point `--work-dir` at the real out-of-repo path:
```bash
bash tools/dist_train.sh ${CFG} 4 \
  --amp --sync_bn torch \
  --work-dir /data/abl/dgf \
  --cfg-options load_from=${LIDAR_CKPT} \
                model.img_backbone.init_cfg.checkpoint=${SWINT_CKPT}
```
Eval on val and record (compare vs the 4×A30 **baseline** number, NOT official):
| metric | baseline (4×A30) | +C (DGFFuser) | Δ |
| --- | --- | --- | --- |
| NDS | `PENDING` | `PENDING` | `PENDING` |
| mAP | `PENDING` | `PENDING` | `PENDING` |

## 6. Memory measurement (PENDING — do not fabricate)
Report peak memory for baseline vs +C under the SAME batch2+amp setting:
```python
import torch; print('peak GiB', torch.cuda.max_memory_allocated()/1024**3)
```
| config | peak mem (GiB) |
| --- | --- |
| baseline | `PENDING` |
| +C (DGFFuser) | `PENDING` |

## 7. all-off integration check (PENDING)
Load the official fusion checkpoint into the **baseline** config model (ConvFuser),
eval on val, and confirm it reproduces the baseline number — proving the +C work
did not perturb the baseline path. Result `PENDING`.

## 8. OOM fallback order (keep effective batch = 32 for comparability)
DGF's full-resolution global attention is the main memory risk.
1. `train_dataloader.batch_size 2 → 1` **and** `optim_wrapper.accumulative_counts 4 → 8`
   (`--cfg-options train_dataloader.batch_size=1 optim_wrapper.accumulative_counts=8`).
2. activation checkpointing: `--cfg-options model.img_backbone.with_cp=True`.
3. **last resort, ask first**: downsample the BEV before DGF attention (deviates
   from the paper) — do NOT do this without approval.

## 9. DGF norm = GroupNorm (default; preserves fg/bg contrast; LayerNorm rejected)
The DGF norm `N` is **GroupNorm(32) by default**, set in `DGFFuser.__init__` (A8).
GroupNorm shares stats across (group-channels × space), so it **preserves the
fg/bg magnitude contrast** the heatmap head needs; and like LN it has no batch
stats / no cross-GPU sync, so it also avoids the SyncBN+fp16 nan (unaffected by
`--sync_bn torch`).
**Why not LayerNorm:** the 850-step smoke REJECTED channel-wise LN — it normalises
each BEV cell to norm √C, forcing `contrast_post≡1.000` (post-norm cell norms all
= √256) and freezing `loss_heatmap` (2.5–2.9, `matched_ious` 0.07). Watch in smoke
(`[DGF-DEBUG contrast]` + `[DGF-DEBUG bug1-isolated]`): the GATE is full
`contrast_post>1`; the isolated `contrast_gn(v_gb)` reads GN alone (BUG-1) even
while BUG-2 is being tamed. If GN ever gives `contrast_post≈1`, reject it.
BN can be forced via `--cfg-options model.fusion_layer.norm_cfg.type=BN2d`
(not recommended — reintroduces the SyncBN/sparse risk).

## 10. Hard offline rules (checklist)
- [ ] No `wget` / `mim download` / online URL in any command or config.
- [ ] `work_dir`, checkpoints, caches all OUTSIDE the source tree.
- [ ] CUDA ops rebuilt after each re-extract (step 0).
- [ ] +C uses the **same** batch2 / accum4 / amp / sync_bn as the baseline.
