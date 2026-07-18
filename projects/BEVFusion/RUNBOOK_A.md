# RUNBOOK — Module A / +A (depth supervision) on the offline 4×A30 server

> Deploy = download the `ablation/depthsup` branch zip → upload → unzip
> (overwrites source tree). **No git, no network on the server.** All result
> numbers below are `PENDING` — fill in after running; **do not fabricate.**

## 0. After every re-extract: rebuild the CUDA ops
The `.so` are git-ignored and live in the source tree, so they vanish on each
re-unzip:
```bash
cd <repo-root>
export CUDA_HOME=$CONDA_PREFIX        # if nvcc came from the conda env
FORCE_CUDA=1 python projects/BEVFusion/setup.py develop
python -c "from projects.BEVFusion.bevfusion.ops import bev_pool, Voxelization; print('ops OK')"
```

## 1. Placeholders to replace (offline, local absolute paths only)
- `projects/BEVFusion/configs/bevfusion_lidar-cam_4xa30_depthsup_nus-3d.py`:
  `work_dir = '/data/abl/depthsup'` → a real absolute path **OUTSIDE the repo**
  (or pass `--work-dir`).
- At launch via `--cfg-options` (LOCAL paths, never URLs):
  - `load_from=<LIDAR_CKPT_PATH>` → local `…bevfusion_lidar…-2628f933.pth`
  - `model.img_backbone.init_cfg.checkpoint=<SWINT_CKPT_PATH>` → local `…swint-nuimages-pretrained.pth`

## 2. CPU unit tests (no GPU/ops needed for the core ones)
```bash
cd <repo-root>
pytest projects/BEVFusion/tests/test_depth_sup.py -q
```
Covers: GT bin discretization, min-pool of nearest non-zero, **sparse mask
(no-point pixels not supervised)**, **softmax-over-bins + BCE on the
probabilities** (activation-matched to the forward LSS lift), the **official
BEVDepth normalization** (per-valid-pixel sum over bins / n_valid), weight
scaling. The `DepthLSSTransform`-level tests (caching only when enabled; zero
new params) run if the ops are importable, else skip.

## 3. all-off / baseline-untouched check (needs compiled ops; CPU fine)
```bash
cd <repo-root>
python tools/check_a_alloff.py
```
Asserts: baseline `use_depth_sup=False`, +A `True`, and that the +A model has
the **same state_dict keys** as the baseline (module A adds 0 parameters).

## 4. Smoke test BEFORE the full run (watch the loss balance!)
```bash
LIDAR_CKPT=<LIDAR_CKPT_PATH>
SWINT_CKPT=<SWINT_CKPT_PATH>
CFG=projects/BEVFusion/configs/bevfusion_lidar-cam_4xa30_depthsup_nus-3d.py

bash tools/dist_train.sh ${CFG} 4 \
  --amp --sync_bn torch \
  --work-dir /data/abl/depthsup_smoke \
  --cfg-options \
    train_cfg.max_epochs=1 \
    load_from=${LIDAR_CKPT} \
    model.img_backbone.init_cfg.checkpoint=${SWINT_CKPT}
```
Confirm in the log:
- a **`loss_depth`** term appears and **trends down**; total loss has **no
  NaN/Inf**;
- **`loss_depth` vs `loss_bbox` are not wildly imbalanced.** With the official
  BEVDepth normalization + weight 3.0, early `loss_depth` lands around ~15-20
  (near-uniform depth probs: per-pixel BCE sum ≈ log(D) + 1 ≈ 5.8, × 3.0) and
  should trend well below that. If it dwarfs the detection losses (and harms
  them), lower the weight:
  `--cfg-options model.view_transform.depth_loss_weight=1.0` (or 0.5);
- the two pretrained checkpoints load (sensible `missing/unexpected_keys`; the
  detection/depth weights already exist in the LiDAR/Swin ckpts as applicable);
- it fits in memory (module A adds **no parameters**; only a small extra
  activation for the depth loss — memory should be ~baseline).

## 5. Full +A training (same recipe as baseline; fairness rule)
Drop `train_cfg.max_epochs=1`, point `--work-dir` at the real out-of-repo path:
```bash
bash tools/dist_train.sh ${CFG} 4 \
  --amp --sync_bn torch \
  --work-dir /data/abl/depthsup \
  --cfg-options load_from=${LIDAR_CKPT} \
                model.img_backbone.init_cfg.checkpoint=${SWINT_CKPT}
```
Eval on val, compare vs the 4×A30 **baseline** number (NOT official):
| metric | baseline (4×A30) | +A (depth-sup) | Δ |
| --- | --- | --- | --- |
| NDS | `PENDING` | `PENDING` | `PENDING` |
| mAP | `PENDING` | `PENDING` | `PENDING` |

## 6. all-off integration check (PENDING)
Load the official fusion checkpoint into the **baseline** config model
(`use_depth_sup` absent → False), eval on val, confirm it reproduces the
baseline number ⇒ the +A work did not perturb the baseline path. Result `PENDING`.

## 7. Tuning note
`depth_loss_weight` (default 3.0 = official BEVDepth) is the main knob. If
detection metrics regress because the depth term dominates, reduce it; if depth
supervision seems to have no effect, the loss may be tiny relative to detection
losses — inspect the logged `loss_depth` magnitude. (Keep batch2/accum4/amp/sync_bn unchanged so +A
stays comparable to the baseline.)

## 8. Hard offline rules (checklist)
- [ ] No `wget` / `mim download` / online URL in any command or config.
- [ ] `work_dir`, checkpoints, caches all OUTSIDE the source tree.
- [ ] CUDA ops rebuilt after each re-extract (step 0).
- [ ] +A uses the **same** batch2 / accum4 / amp / sync_bn as the baseline.
