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
Eval on val, compare vs the 4×A30 **baseline** number (NOT official). Paired
6-epoch results (same seed 577127641, best epoch = ep5 for all):
| metric | baseline | +A w1.0 | +A w3.0 |
| --- | --- | --- | --- |
| NDS | 0.7060 | 0.7034 (−0.26) | 0.6985 (−0.75) |
| mAP | 0.6648 | 0.6600 (−0.48) | 0.6508 (−1.40) |

Depth supervision is **net-negative at every weight** and monotonically worse
with weight ⇒ weight-tuning cannot make it help (floor = baseline). See §5b.

## 5b. Full +A v2 training (input-dropout, closes the leakage shortcut)
Same command as §5 but launch from the v2 config (adds
`depth_input_keep_ratio=0.3`), or inject the key via `--cfg-options` on the
stock 6e config used for the baseline (recommended — keeps the paired recipe):
```bash
CFG=projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py
bash tools/dist_train.sh ${CFG} 4 \
  --amp --sync_bn torch \
  --work-dir work_dirs/bevfusion_lidar-cam_official6e_depthsup_v2_seed577127641 \
  --cfg-options \
    randomness.seed=577127641 \
    train_dataloader.batch_size=2 \
    optim_wrapper.type=AmpOptimWrapper \
    optim_wrapper.loss_scale=512.0 \
    optim_wrapper.accumulative_counts=4 \
    load_from=${LIDAR_CKPT} \
    model.img_backbone.init_cfg.checkpoint=${SWINT_CKPT} \
    model.view_transform.use_depth_sup=True \
    model.view_transform.depth_loss_weight=3.0 \
    model.view_transform.depth_input_keep_ratio=0.3
```
Same seed/recipe as baseline+w3.0, so v2 is paired to both; the only change vs
w3.0 is the train-time input dropout. Expect `loss_depth` to start HIGHER than
w3.0 (the completion task is harder) and `grad_norm` similar-to-lower than w3.0.
| metric | baseline | +A w3.0 | +A v2 (keep 0.3) |
| --- | --- | --- | --- |
| NDS | 0.7060 | 0.6985 | `PENDING` |
| mAP | 0.6648 | 0.6508 | `PENDING` |

**v3 strict no-overlap** (optional, stronger): add
`model.view_transform.depth_loss_heldout_only=True` to the `--cfg-options` above
to supervise ONLY held-out cells (no copy possible at all). v2 still lets the
retained ~30% of cells be conditionally copied; v3 removes that.

**Two orthogonal axes — do NOT conflate.** v2/v3 fix input-side leakage only;
the auxiliary-loss magnitude is separate. Weight 3.0 above pairs v2/v3 to +A
w3.0 (only change = the fix), but the dose-response (w1.0 −0.26 / w3.0 −0.75
NDS) means weight still matters independently. So for v2 (and v3 if run) ALSO do
a `depth_loss_weight=1.0` arm, and in EVERY run watch:
- `loss_depth` vs `loss_bbox` (depth must not dwarf detection), and
- total `grad_norm` (v1 baseline ~0.9; w1.0 peaked ~9; w3.0 ~80 — a sign the
  depth term is over-driving the shared trunk).

**Shortcut probe** (before/after, read-only, needs GPU+val+ops):
```bash
python projects/BEVFusion/tools/probe_depth_shortcut.py \
  --config ${CFG} --checkpoint <ckpt>.pth --num-batches 50 --occlude 0.5 \
  --out outputs/probe_<name>.json
```
Run on baseline, +A w3.0, and +A v2. A large kept-minus-held-out accuracy gap =
copy/shortcut; v2 should SHRINK that gap vs w3.0.

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
