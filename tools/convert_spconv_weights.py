# Copyright (c) OpenMMLab. All rights reserved.
"""Convert spconv 1.x sparse-conv weights to the spconv 2.x kernel layout.

The official BEVFusion checkpoints released by OpenMMLab were trained/saved
with **spconv 1.x**, whose 3D sparse-conv kernels are laid out as::

    [out_ch, kx, ky, kz, in_ch]            # spconv 1.x (official checkpoints)

while an environment running **spconv 2.x** expects::

    [kx, ky, kz, in_ch, out_ch]            # spconv 2.x (this model)

When the 1.x weights are loaded into a 2.x model every parameter under
``pts_middle_encoder.*`` (the whole 3D sparse backbone) size-mismatches.
Because ``load_from`` is non-strict those parameters are silently dropped and
re-initialised randomly, which collapses detection quality (e.g. NDS ~0.007
instead of ~0.696 for the LiDAR-only checkpoint).

This script rewrites every affected 5D kernel with::

    new_w = w.permute(1, 2, 3, 4, 0).contiguous()   # [O,kx,ky,kz,I] -> [kx,ky,kz,I,O]

and saves a *new* checkpoint (the input is never overwritten).

Robustness
----------
The conversion is **shape-driven**, never blind: a reference model is built
from the config and the target shape of every parameter is read from it. For
each checkpoint parameter we then decide:

* already matches the reference shape          -> keep as-is (idempotent re-run);
* 5D and ``permute(1,2,3,4,0)`` makes it match  -> permute;
* not present in the reference model           -> pass through untouched
  (e.g. the fusion checkpoint's ``img_backbone``/``view_transform``/
  ``fusion_layer`` weights, which have no spconv layout issue);
* otherwise                                     -> recorded as *unresolved*
  (never force-converted; the script exits non-zero).

Why ``state_dict._metadata`` is preserved
-----------------------------------------
mmdet3d ships an auto-convert hook
(``mmdet3d/models/layers/spconv/overwrite_spconv/write_spconv2.py``) that
permutes 1.x->2.x weights on load *only when* a module's
``state_dict._metadata['version'] != 2``. The official checkpoints already
carry ``version == 2`` (which is exactly why the hook skips them and the raw
1.x bytes mismatch). After we fix the layout we keep ``_metadata`` intact so
the hook continues to skip these modules and does **not** re-permute the
now-correct weights when the converted checkpoint is loaded.

Ground truth (run by the user on a GPU server -- not in this step)
------------------------------------------------------------------
Evaluate the converted ``...-2628f933-spconv2.pth`` with the LiDAR config; it
should give **NDS ~0.696 / mAP ~0.649**. If the result is far from that, the
permute rule may need an extra spatial-dim flip for some spconv versions'
inverse/down-sample kernels -- but for the standard ``SubMConv3d`` /
``SparseConv3d`` layers used by this backbone, ``permute(1,2,3,4,0)`` is the
standard and sufficient transform.

Examples
--------
LiDAR-only::

    python tools/convert_spconv_weights.py \\
      checkpoints/..._nus-3d-2628f933.pth \\
      --out checkpoints/..._nus-3d-2628f933-spconv2.pth \\
      --config projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py

LiDAR+camera fusion (img/view/fusion weights pass through untouched)::

    python tools/convert_spconv_weights.py \\
      checkpoints/..._nus-3d-5239b1af.pth \\
      --out checkpoints/..._nus-3d-5239b1af-spconv2.pth \\
      --config projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py

Run a CPU-only self test that needs neither the real weights nor a network::

    python tools/convert_spconv_weights.py --self-test
"""
import argparse
import os.path as osp
import sys
import tempfile
from collections import OrderedDict

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.utils import import_modules_from_strings

from mmdet3d.registry import MODELS

# Default reference config (pure LiDAR). The LiDAR backbone is identical in the
# fusion checkpoint, so the same reference works for both checkpoints.
DEFAULT_CONFIG = (
    'projects/BEVFusion/configs/'
    'bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py')

# Prefix of the parameters affected by the spconv layout difference.
SPCONV_PREFIX = 'pts_middle_encoder.'

# permute used for the 1.x -> 2.x conversion: [O,kx,ky,kz,I] -> [kx,ky,kz,I,O].
PERMUTE = (1, 2, 3, 4, 0)
# inverse permute, only used by --self-test to fabricate a fake 1.x checkpoint.
INV_PERMUTE = (4, 0, 1, 2, 3)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert spconv 1.x sparse-conv weights to spconv 2.x '
        'kernel layout for BEVFusion checkpoints.')
    parser.add_argument(
        'in_ckpt',
        nargs='?',
        default=None,
        help='input checkpoint (spconv 1.x layout). Not required with '
        '--self-test.')
    parser.add_argument(
        '--out',
        default=None,
        help='output checkpoint path. If omitted, a "-spconv2" suffix is '
        'inserted before the extension of the input path.')
    parser.add_argument(
        '--config',
        default=DEFAULT_CONFIG,
        help='reference config used to build the model and read target '
        f'parameter shapes (default: {DEFAULT_CONFIG}). Run from the repo '
        'root so that "projects.BEVFusion..." is importable.')
    parser.add_argument(
        '--self-test',
        action='store_true',
        help='run a CPU-only round-trip self test (no real weights, no '
        'network) and exit.')
    return parser.parse_args()


def build_reference_model(config_path):
    """Build the model offline from a config to read target parameter shapes.

    No Runner, no data and no weights are involved. Returns the built model so
    callers can also read ``state_dict()._metadata`` from it if needed.
    """
    cfg = Config.fromfile(config_path)
    # Register custom modules (e.g. projects.BEVFusion.bevfusion) declared in
    # the config BEFORE initialising the default scope / building the model.
    custom_imports = cfg.get('custom_imports', None)
    if custom_imports:
        import_modules_from_strings(**custom_imports)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    model = MODELS.build(cfg.model)
    return model


def derive_out_path(in_path):
    """Insert a ``-spconv2`` suffix before the extension of ``in_path``."""
    root, ext = osp.splitext(in_path)
    return f'{root}-spconv2{ext or ".pth"}'


def torch_load(path):
    """Load a checkpoint on CPU, tolerant of torch>=2.6 ``weights_only``.

    Full mmengine checkpoints contain non-tensor objects (``meta``,
    ``message_hub`` ...) that cannot be unpickled with the new
    ``weights_only=True`` default, so fall back to ``weights_only=False``.
    """
    try:
        return torch.load(path, map_location='cpu')
    except Exception:
        # On torch<1.13 the default load above already succeeded, so reaching
        # here implies the kwarg exists.
        return torch.load(path, map_location='cpu', weights_only=False)


def split_checkpoint(ckpt):
    """Return ``(state_dict, has_state_dict_key)``.

    Supports both a full checkpoint (``{'state_dict': ...}``) and a bare
    state_dict at the top level.
    """
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        return ckpt['state_dict'], True
    return ckpt, False


def convert_state_dict(state_dict, ref_shapes):
    """Convert a state_dict to the 2.x layout, driven by ``ref_shapes``.

    Returns a tuple ``(new_state_dict, permuted, passed_through, unresolved,
    missing)`` where:

    * ``permuted``       -- list of ``(name, old_shape, new_shape)``;
    * ``passed_through`` -- list of names kept unchanged (already matching or
      not present in the reference model);
    * ``unresolved``     -- list of ``(name, ckpt_shape, ref_shape)`` that are
      in the reference model but neither match nor can be fixed by permute;
    * ``missing``        -- list of names present in the reference model but
      absent from the checkpoint (informational only).
    """
    new_state_dict = OrderedDict()
    permuted = []
    passed_through = []
    unresolved = []

    for name, weight in state_dict.items():
        ref_shape = ref_shapes.get(name, None)

        if ref_shape is None:
            # Not part of the reference model (e.g. fusion-only branches).
            new_state_dict[name] = weight
            passed_through.append(name)
            continue

        cur_shape = tuple(weight.shape)
        if cur_shape == ref_shape:
            # Already in the target layout -> idempotent.
            new_state_dict[name] = weight
            passed_through.append(name)
            continue

        if weight.dim() == 5 and tuple(
                weight.permute(*PERMUTE).shape) == ref_shape:
            new_weight = weight.permute(*PERMUTE).contiguous()
            new_state_dict[name] = new_weight
            permuted.append((name, cur_shape, tuple(new_weight.shape)))
            continue

        # Neither matches nor is fixable by the standard permute.
        new_state_dict[name] = weight
        unresolved.append((name, cur_shape, ref_shape))

    # Preserve per-module metadata (carries spconv ``version``) so the on-load
    # auto-convert hook does not re-permute the now-correct weights.
    metadata = getattr(state_dict, '_metadata', None)
    if metadata is not None:
        new_state_dict._metadata = metadata

    ckpt_names = set(state_dict.keys())
    missing = [name for name in ref_shapes if name not in ckpt_names]

    return new_state_dict, permuted, passed_through, unresolved, missing


def recheck(new_state_dict, ref_shapes):
    """Independently re-verify shapes after conversion.

    Returns ``(total_mismatches, spconv_mismatches)`` counting parameters that
    are present in BOTH the converted state_dict and the reference model but
    whose shapes still differ.
    """
    total = 0
    spconv = 0
    for name, weight in new_state_dict.items():
        ref_shape = ref_shapes.get(name, None)
        if ref_shape is None:
            continue
        if tuple(weight.shape) != ref_shape:
            total += 1
            if name.startswith(SPCONV_PREFIX):
                spconv += 1
    return total, spconv


def report_and_gate(permuted, passed_through, unresolved, missing,
                    new_state_dict, ref_shapes):
    """Print the verification report and return a process exit code."""
    print('\n=== Permuted parameters (spconv 1.x -> 2.x) ===')
    if permuted:
        for name, old_shape, new_shape in permuted:
            print(f'  {name}: {tuple(old_shape)} -> {tuple(new_shape)}')
    else:
        print('  (none)')

    print('\n=== Counts ===')
    print(f'  permuted       : {len(permuted)}')
    print(f'  passed-through : {len(passed_through)} '
          '(already-matching + not-in-reference)')
    print(f'  unresolved     : {len(unresolved)}')

    if missing:
        print(f'\n=== WARNING: {len(missing)} reference param(s) missing from '
              'the checkpoint ===')
        for name in missing:
            print(f'  {name}')

    total_mismatch, spconv_mismatch = recheck(new_state_dict, ref_shapes)
    print('\n=== Post-conversion re-check (params present in both) ===')
    print(f'  total shape mismatches        : {total_mismatch}')
    print(f'  {SPCONV_PREFIX}* shape mismatches : {spconv_mismatch}')

    if unresolved:
        print('\n=== ERROR: unresolved parameters (NOT converted) ===')
        for name, cur_shape, ref_shape in unresolved:
            print(f'  {name}: checkpoint {tuple(cur_shape)} vs reference '
                  f'{tuple(ref_shape)}')
        print('\nConversion FAILED: unresolved > 0. The permute rule did not '
              'cover every affected parameter; not pretending success.')
        return 1

    # Hard acceptance: every shared parameter (and in particular the whole
    # spconv backbone) must match the reference exactly.
    assert spconv_mismatch == 0, (
        f'{SPCONV_PREFIX}* still has {spconv_mismatch} size mismatch(es) '
        'after conversion')
    assert total_mismatch == 0, (
        f'{total_mismatch} reference parameter(s) still mismatch after '
        'conversion')
    print('\nConversion OK: all reference parameters match; '
          f'{SPCONV_PREFIX}* size mismatch = 0; unresolved = 0.')
    return 0


def convert(in_ckpt, out_ckpt, config_path):
    """Convert ``in_ckpt`` to 2.x layout and write ``out_ckpt``."""
    if osp.realpath(in_ckpt) == osp.realpath(out_ckpt):
        raise ValueError(
            f'Refusing to overwrite the input checkpoint: {in_ckpt}')

    print(f'Building reference model from: {config_path}')
    model = build_reference_model(config_path)
    ref_shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}

    print(f'Loading checkpoint: {in_ckpt}')
    ckpt = torch_load(in_ckpt)
    state_dict, has_sd_key = split_checkpoint(ckpt)

    new_state_dict, permuted, passed_through, unresolved, missing = \
        convert_state_dict(state_dict, ref_shapes)

    code = report_and_gate(permuted, passed_through, unresolved, missing,
                           new_state_dict, ref_shapes)
    if code != 0:
        sys.exit(code)

    # Write back, preserving every other top-level key (meta, message_hub ...).
    if has_sd_key:
        ckpt['state_dict'] = new_state_dict
        out_obj = ckpt
    else:
        out_obj = new_state_dict

    torch.save(out_obj, out_ckpt)
    print(f'\nSaved converted checkpoint to: {out_ckpt}')


def self_test(config_path):
    """CPU-only round-trip test: fabricate a fake 1.x ckpt and convert it.

    Builds the reference model, makes a synthetic "spconv 1.x" checkpoint by
    inverse-permuting every 5D kernel, runs the conversion and asserts that the
    result matches the reference shapes and that a second pass is a no-op.
    Needs neither the real weights nor any network access.
    """
    print(f'[self-test] Building reference model from: {config_path}')
    model = build_reference_model(config_path)
    ref_sd = model.state_dict()
    ref_shapes = {k: tuple(v.shape) for k, v in ref_sd.items()}

    # Fabricate a fake spconv 1.x state_dict (inverse layout for 5D kernels).
    fake_1x = OrderedDict()
    n_kernels = 0
    for name, weight in ref_sd.items():
        if weight.dim() == 5:
            fake_1x[name] = weight.permute(*INV_PERMUTE).contiguous()
            n_kernels += 1
        else:
            fake_1x[name] = weight.clone()
    # Mimic official checkpoints: spconv modules carry version==2 metadata.
    fake_1x._metadata = getattr(ref_sd, '_metadata', None)
    print(f'[self-test] fabricated fake-1.x checkpoint with {n_kernels} '
          '5D kernels inverse-permuted')
    assert n_kernels > 0, 'reference model has no 5D sparse-conv kernels'

    with tempfile.TemporaryDirectory() as tmp:
        in_path = osp.join(tmp, 'fake_1x.pth')
        out_path = osp.join(tmp, 'fake_1x-spconv2.pth')
        out_path2 = osp.join(tmp, 'fake_1x-spconv2-again.pth')
        torch.save({'state_dict': fake_1x, 'meta': {'note': 'self-test'}},
                   in_path)

        # First pass: should permute exactly the fabricated kernels.
        ckpt = torch_load(in_path)
        sd, _ = split_checkpoint(ckpt)
        # The reloaded state_dict must still carry the metadata we attached.
        assert getattr(sd, '_metadata', None) is not None, \
            '_metadata did not survive save/load'
        new_sd, permuted, _, unresolved, missing = convert_state_dict(
            sd, ref_shapes)
        torch.save({'state_dict': new_sd}, out_path)

        assert len(unresolved) == 0, f'unresolved: {unresolved}'
        assert len(permuted) == n_kernels, (
            f'expected {n_kernels} permutes, got {len(permuted)}')
        assert not [m for m in missing if m.startswith(SPCONV_PREFIX)], (
            f'spconv params missing: {missing}')
        total_mismatch, spconv_mismatch = recheck(new_sd, ref_shapes)
        assert total_mismatch == 0 and spconv_mismatch == 0, (
            f'post-conversion mismatch total={total_mismatch} '
            f'spconv={spconv_mismatch}')
        # convert_state_dict must preserve its input's _metadata (same object),
        # so the on-load hook keeps skipping these modules.
        assert getattr(new_sd, '_metadata', None) is getattr(
            sd, '_metadata', None), '_metadata not preserved by conversion'

        # Second pass on the converted file must be a no-op (idempotent).
        sd2, _ = split_checkpoint(torch_load(out_path))
        new_sd2, permuted2, _, unresolved2, _ = convert_state_dict(
            sd2, ref_shapes)
        torch.save({'state_dict': new_sd2}, out_path2)
        assert len(permuted2) == 0, (
            f'second pass not idempotent, permuted {len(permuted2)}')
        assert len(unresolved2) == 0, f'second-pass unresolved: {unresolved2}'

    print('[self-test] PASS: 1.x->2.x conversion is correct, shape-exact, '
          'metadata-preserving and idempotent.')


def main():
    args = parse_args()

    if args.self_test:
        self_test(args.config)
        return

    if args.in_ckpt is None:
        raise SystemExit(
            'error: in_ckpt is required (or pass --self-test). '
            'See --help for usage.')

    out_ckpt = args.out if args.out is not None else derive_out_path(
        args.in_ckpt)
    convert(args.in_ckpt, out_ckpt, args.config)


if __name__ == '__main__':
    main()
