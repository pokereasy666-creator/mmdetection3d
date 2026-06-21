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
resolution-agnostic, and the use_out_proj / GroupNorm options.

## 3. Baseline-untouched check (needs compiled ops; CPU is fine)
```bash
cd <repo-root>
python tools/check_c_alloff.py
```
Asserts baseline=ConvFuser, +C=DGFFuser, and that **only** `fusion_layer.*`
keys change (every other parameter identical) ⇒ module C touches nothing else.

## 4. Smoke test BEFORE the full run (catch OOM / NaN / bad ckpt-load early)
Run a few dozen iters (or 1 epoch) and watch the log:
```bash
LIDAR_CKPT=<LIDAR_CKPT_PATH>
SWINT_CKPT=<SWINT_CKPT_PATH>
CFG=projects/BEVFusion/configs/bevfusion_lidar-cam_4xa30_dgf_nus-3d.py

bash tools/dist_train.sh ${CFG} 4 \
  --amp --sync_bn torch \
  --work-dir /data/abl/dgf_smoke \
  --cfg-options \
    train_cfg.max_epochs=1 \
    load_from=${LIDAR_CKPT} \
    model.img_backbone.init_cfg.checkpoint=${SWINT_CKPT}
```
Confirm in the log:
- **weights loaded**: the `load_from` line + sensible `missing_keys` /
  `unexpected_keys` (the new `fusion_layer.*` DGF weights are expected to be in
  `missing_keys` w.r.t. the LiDAR-only checkpoint — that's normal, they train
  from scratch);
- **forward+backward OK**, loss is finite and **trends down**, **no NaN/Inf**;
- **memory fits** (no CUDA OOM) — note the peak (step 6).

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

## 9. DGF norm = GroupNorm (default; was the NaN root cause)
The DGF norm is **GroupNorm by default** (`num_groups=32`), set in
`DGFFuser.__init__` [module-C/bn-to-groupnorm]. BatchNorm2d was the NaN root
cause: `norm1` runs on the sparse, low-variance `lidar_proj(lidar_bev)` output
(std~0.056) and BN renormalises that std to ~1, amplifying the feature norm ~18×
(229→~4218) → `grad_norm=NaN`, `loss_heatmap` explodes (diagnosed via the
DGF_DEBUG step-norms). GroupNorm is batch-independent and unaffected by
`--sync_bn torch`. To force BN back (not recommended):
`--cfg-options model.fusion_layer.norm_cfg.type=BN2d`.

## 10. Hard offline rules (checklist)
- [ ] No `wget` / `mim download` / online URL in any command or config.
- [ ] `work_dir`, checkpoints, caches all OUTSIDE the source tree.
- [ ] CUDA ops rebuilt after each re-extract (step 0).
- [ ] +C uses the **same** batch2 / accum4 / amp / sync_bn as the baseline.
