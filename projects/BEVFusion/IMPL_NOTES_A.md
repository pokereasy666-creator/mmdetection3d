# IMPL_NOTES — Module A / Explicit Depth Supervision (BEVDepth-style)

> Module A adds **BEVDepth (arXiv:2206.10092) explicit depth supervision** to
> the BEVFusion baseline, which (per `PROBE_A_depth.md`) has **no** depth loss.
> Switched by `use_depth_sup` (default False). With it off, the baseline is
> **byte-identical** and gains **zero** parameters.

## What changed (only these; baseline logic otherwise untouched)
| File | Change | Guarded? |
| --- | --- | --- |
| `bevfusion/depth_sup.py` | **NEW** torch-only helpers: `downsample_gt_depth`, `depth_bce_loss`. | n/a (only used when on) |
| `bevfusion/depth_lss.py` | `DepthLSSTransform.__init__` gains `use_depth_sup=False`, `depth_loss_weight=0.5`; `get_cam_feats` stashes logits+GT; new `get_depth_loss`. | yes (`if self.use_depth_sup`) |
| `bevfusion/bevfusion.py` | `loss()` adds `loss_depth` only when the view transform's `use_depth_sup` is True. | yes (`getattr(..., False)`) |
| `configs/bevfusion_lidar-cam_4xa30_depthsup_nus-3d.py` | **NEW** +A config (inherits 4xA30 baseline, sets the flag, out-of-repo work_dir). | n/a |

## Switch carrier = module attributes (not a changed return signature)
Per the design rule, the depth tensors are surfaced via **plain attributes** on
the view transform, set only when enabled:
- `DepthLSSTransform._depth_pred_logits` — pre-softmax logits `(B*N, D, fH, fW)`;
- `DepthLSSTransform._depth_gt` — sparse LiDAR depth `(B*N, 1, iH, iW)`.

`get_cam_feats` / `view_transform` / `extract_img_feat` / `extract_feat`
**return signatures are unchanged**. `BEVFusion.loss` reads the attributes via
`self.view_transform` after `extract_feat`, then calls `get_depth_loss()` (which
also clears the caches). This means the all-off code path is exactly the
baseline's — see the equivalence argument below.

### all-off byte-identical equivalence (the铁律)
With `use_depth_sup=False`:
- `get_cam_feats` runs the original line `depth = x[:, :self.D].softmax(dim=1)`
  (the `else` branch); the `if self.use_depth_sup` blocks are skipped, so **no
  tensor is cached** and the produced `x` is identical to the baseline.
- `BEVFusion.loss`: `getattr(vt, 'use_depth_sup', False)` is False ⇒ **no
  `loss_depth` key** ⇒ the loss dict equals the baseline's.
- The extra `__init__` attributes (`use_depth_sup`, `depth_loss_weight`,
  `_depth_*`) are **not** `nn.Parameter`/buffers ⇒ they do not appear in
  `parameters()`/`state_dict()`. Module A adds **0** parameters, so the +A model
  has the **same `state_dict` keys** as the baseline (verified by
  `tools/check_a_alloff.py`). The only train-time difference is the loss term.

## GT downsample + binning (`downsample_gt_depth`)
Re-implements BEVDepth's `get_downsampled_gt_depth`, self-contained:
1. Reshape `(M,1,iH,iW)` into `(ds_h x ds_w)` image patches per feature cell
   (`ds = iH//fH = 256//32 = 8`).
2. **Min-pool the nearest NON-zero depth** in each patch (zeros → `1e5` so they
   lose the `min`) — avoids the sparse zeros polluting the GT.
3. Discretize per `dbound=[d_min,d_max,d_step]`: `bin = floor((d-d_min)/d_step)`,
   `D = (d_max-d_min)/d_step = 118` bins; one-hot.
4. `valid = (d >= d_min) & (d < d_max)` — True only where a LiDAR point lands in
   range. **This is the sparse mask: no-point pixels are never supervised.**

## Depth loss (`depth_bce_loss`)
- **softmax over the depth-bin dim (dim=1) + BCE on probabilities**, computed
  **only over `valid` pixels**. The softmax is the SAME activation the forward
  LSS uses (`DepthLSSTransform.get_cam_feats`: `logits.softmax(dim=1)`), so the
  supervision matches the forward lift. ([module-A/fix-softmax-bce] — previously
  per-bin `binary_cross_entropy_with_logits`, which mismatched the forward
  softmax and diluted the signal.)
- `F.binary_cross_entropy` is NOT autocast-safe, so it runs inside
  `torch.cuda.amp.autocast(enabled=False)` with fp32 inputs (`logits.float()`,
  `one_hot.float()`) to avoid fp16 instability.
- No valid pixel in a batch ⇒ returns a graph-connected `0` (no NaN).
- Weighted by `depth_loss_weight` (config, default `0.5`).
- Added to total loss as `losses['loss_depth']`.

## ASSUMPTIONs (paper/instruction-unspecified → chosen value + reason)
| ID | Decision | Reason |
| --- | --- | --- |
| **A-D1** | Loss = **softmax over bins (dim=1) + `F.binary_cross_entropy` on probs**, in fp32 under `autocast(enabled=False)`. | [module-A/fix-softmax-bce] Activation matched to the forward LSS lift (`logits.softmax(dim=1)`); the earlier per-bin `binary_cross_entropy_with_logits` used independent sigmoids that mismatched the forward softmax and diluted supervision. `binary_cross_entropy` needs fp32 (not autocast-safe). |
| **A-D2** | Bin index = `floor((d - d_min)/d_step)`, range `[0, D-1]` (clamped); `valid = d∈[d_min,d_max)`. | Simple, deterministic mapping over `dbound`; out-of-range LiDAR points are masked out (not clamped into edge bins for supervision). |
| **A-D3** | Downsample = **min-pool of nearest non-zero** over the `8x8` patch. | BEVDepth convention; keeps the closest real depth, ignores empty pixels. |
| **A-D4** | `depth_loss_weight` default **0.5**. | Task-specified default; a common BEVDepth-range weight. Tune down if `loss_depth` dwarfs `loss_bbox` (RUNBOOK §3). |
| **A-D5** | Logits source = `depthnet` output channels `[:D]` (`depth_lss.py`), captured **before** the softmax at the (former) `:416` line. | `PROBE_A_depth.md` Q3: that is the predicted depth distribution; BCE needs the pre-softmax logits. |
| **A-D6** | Switch + weight live on the **view transform** (`DepthLSSTransform`); `BEVFusion.loss` reads them via `getattr(self.view_transform, ...)`. | Keeps all depth config in one place; BEVFusion change stays a minimal guarded read. |
| **A-D7** | Depth loss computed in the resolution the logits are produced at (`feature_size=[32,88]`), GT downsampled to match. | Logits are at feature resolution; matches BEVDepth. |
| **A-D8** | Caches are read-and-cleared in `get_depth_loss`; during `predict()` they are set but unused (overwritten next forward). | Avoid stale references / memory growth. |

## Out of scope (intentionally NOT done)
- Modules C / D; any change to baseline logic beyond the guarded hooks above.
- Any number (loss magnitude / mAP / NDS / GB) — all PENDING, measured on the
  4xA30 server (see RUNBOOK_A.md).
