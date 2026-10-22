# -*- coding: utf-8 -*-
"""Sub-cell position estimates from the 3x3 heatmap neighbourhood of a peak.

Shared by the exec doc #3 step 0a analysis and by method M-a (one
implementation, so the offline numbers and the trained method decode the
same way). Pure torch, no mmengine / mmdet3d import.

Conventions (TransFusionHead of projects/BEVFusion, audit T3 / T4):
  heat is [B, C, H, W]; a flat cell index is row * W + col; the cell
  centre in grid units is (row + 0.5, col + 0.5). Offsets are returned in
  grid units as (d_row, d_col); the caller maps rows / cols to x / y (row is
  x unless bev_pos is transposed).
"""
import torch

OK, BORDER, NOT_NEG_DEF = 0, 1, 2
STATUS_NAMES = {OK: 'ok', BORDER: 'border', NOT_NEG_DEF: 'not_neg_def'}


def neighborhood3x3(heat, idx, cls):
    """3x3 values of channel cls around cell idx.

    Args:
        heat: [B, C, H, W] heatmap (sigmoid probabilities).
        idx:  [B, P] flat cell index (row * W + col).
        cls:  [B, P] class channel of each peak.
    Returns:
        nb     [B, P, 3, 3]  nb[..., i, j] at (row + i - 1, col + j - 1)
        valid  [B, P, 3, 3]  False where the neighbour is off the map (the
                             value there is a clamped duplicate)
        border [B, P]        True if any neighbour is off the map
    """
    B, C, H, W = heat.shape
    P = idx.shape[1]
    r, c = idx // W, idx % W
    d = torch.arange(-1, 2, device=idx.device)
    rr = r[..., None, None] + d.view(1, 1, 3, 1)
    cc = c[..., None, None] + d.view(1, 1, 1, 3)
    valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    lin = cls[..., None, None] * (H * W) + rr.clamp(0, H - 1) * W + \
        cc.clamp(0, W - 1)
    nb = heat.reshape(B, C * H * W).gather(1, lin.reshape(B, -1))
    border = ~valid.reshape(B, P, 9).all(-1)
    return nb.reshape(B, P, 3, 3), valid, border


def dark_offsets(nb, border, clamp=0.5, eps=1e-12):
    """DARK: Newton step on log(heat) from finite differences.

    offset = -H^-1 g with central-difference gradient g and Hessian H
    (incl. the cross term), each component clipped to +-clamp. Falls back to
    the cell centre (offset 0) on the map border and wherever H is not
    negative definite (singular, indefinite or positive semi-definite).
    Computed in float64.

    Returns:
        off     [B, P, 2] float32 (d_row, d_col), grid units
        status  [B, P] long: OK / BORDER / NOT_NEG_DEF
        clipped [B, P] bool: an OK offset was clipped to +-clamp
    """
    L = nb.double().clamp_min(eps).log()
    g_r = 0.5 * (L[..., 2, 1] - L[..., 0, 1])
    g_c = 0.5 * (L[..., 1, 2] - L[..., 1, 0])
    h_rr = L[..., 2, 1] - 2.0 * L[..., 1, 1] + L[..., 0, 1]
    h_cc = L[..., 1, 2] - 2.0 * L[..., 1, 1] + L[..., 1, 0]
    h_rc = 0.25 * (L[..., 2, 2] - L[..., 2, 0] - L[..., 0, 2] + L[..., 0, 0])
    det = h_rr * h_cc - h_rc * h_rc
    ok = (~border) & (h_rr < 0) & (det > 0)
    safe = torch.where(ok, det, torch.ones_like(det))
    d_r = -(h_cc * g_r - h_rc * g_c) / safe
    d_c = -(h_rr * g_c - h_rc * g_r) / safe
    off = torch.stack([d_r, d_c], dim=-1)
    off = torch.where(ok[..., None], off, torch.zeros_like(off))
    clipped = ok & (off.abs() > clamp).any(-1)
    off = off.clamp(-clamp, clamp)
    status = torch.full_like(border, NOT_NEG_DEF, dtype=torch.long)
    status[ok] = OK
    status[border] = BORDER
    return off.float(), status, clipped


def softargmax_offsets(nb, valid):
    """Probability-weighted centre of the (on-map part of the) 3x3 window.

    Returns off [B, P, 2] float32 (d_row, d_col), grid units.
    """
    w = nb.double() * valid.double()
    d = torch.arange(-1, 2, device=nb.device, dtype=torch.float64)
    s = w.sum((-1, -2)).clamp_min(1e-30)
    d_r = (w * d.view(3, 1)).sum((-1, -2)) / s
    d_c = (w * d.view(1, 3)).sum((-1, -2)) / s
    return torch.stack([d_r, d_c], dim=-1).float()


def rowcol_to_xy(off_rc, transposed=False):
    """(d_row, d_col) -> (d_x, d_y) in grid units."""
    return off_rc.flip(-1) if transposed else off_rc


def bilinear_gather(feat_flat, pos_xy, H, W):
    """Sample [B, C, H*W] features at continuous grid positions.

    pos_xy [B, P, 2] is in bev_pos units (x = row + 0.5, y = col + 0.5).
    Written with four gathers so that a position exactly on a cell centre
    (fractional part 0) returns the plain gather of that cell bit for bit:
    the other three weights are exactly 0 (exec doc #3 S-M0). Positions are
    clamped to the map; gradients flow into feat_flat, not into pos_xy.
    Returns [B, C, P] in feat_flat's dtype.
    """
    B, C, _ = feat_flat.shape
    pos = pos_xy.detach().float()
    r = (pos[..., 0] - 0.5).clamp(0, H - 1)
    c = (pos[..., 1] - 0.5).clamp(0, W - 1)
    r0, c0 = r.floor(), c.floor()
    fr, fc = r - r0, c - c0
    r0, c0 = r0.long(), c0.long()
    r1, c1 = (r0 + 1).clamp(max=H - 1), (c0 + 1).clamp(max=W - 1)

    def take(rr, cc):
        idx = (rr * W + cc)[:, None, :].expand(-1, C, -1)
        return feat_flat.gather(2, idx).float()

    out = take(r0, c0) * ((1 - fr) * (1 - fc))[:, None] + \
        take(r0, c1) * ((1 - fr) * fc)[:, None] + \
        take(r1, c0) * (fr * (1 - fc))[:, None] + \
        take(r1, c1) * (fr * fc)[:, None]
    return out.to(feat_flat.dtype)
