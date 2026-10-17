#!/usr/bin/env bash
# One-time, READ-ONLY copy of the exec doc #1 inputs out of the EP-Fusion
# repo into this workspace (decision A-1). Nothing is written to, created
# in, or run inside the EP-Fusion repo: it is only read with cp / md5sum /
# readlink. Run from the root of the Dome-DETR diagnosis workspace.
#
#   bash tools/query_diag/exec01_fetch_inputs.sh
#
# Env: EP_REPO (default below), INPUTS (default work_dirs/exec01_inputs)
# Copies (md5 verified):  epoch_5.pth, swint-nuimages-pretrained.pth, the
#   training run's dumped config(s), and per timestamp dir its log and
#   vis_data/scalars.json (val metrics, used by S0d).
# Links: data/nuscenes -> the real nuScenes directory used by EP-Fusion,
#   created in THIS workspace only if it does not exist yet.
set -euo pipefail

EP_REPO=${EP_REPO:-/212022085500129/bevfusion-concat/mmdetection3d-claude-jolly-wozniak-c4YMQ}
RUN_NAME=bevfusion_lidar-cam_official6e_4xa30_amp512_accum4_seed577127641
CKPT_NAME=epoch_5.pth
SWIN=checkpoints/swint-nuimages-pretrained.pth
DEST=${INPUTS:-$PWD/work_dirs/exec01_inputs}

die() { echo "[fetch] $*" >&2; exit 2; }
under() { case "$(readlink -m -- "$1")/" in "$(readlink -m -- "$2")"/*) return 0 ;; esac; return 1; }

[ -d tools/query_diag ] && [ -d projects/BEVFusion ] || \
  die "run from the Dome-DETR workspace root (cwd: $PWD)"
if under "$PWD" "$EP_REPO"; then die "cwd is inside the EP-Fusion repo"; fi
if under "$DEST" "$EP_REPO"; then die "INPUTS=$DEST is inside the EP-Fusion repo"; fi
SRC_RUN=$EP_REPO/work_dirs/$RUN_NAME
[ -f "$SRC_RUN/$CKPT_NAME" ] || die "missing $SRC_RUN/$CKPT_NAME"
[ -f "$EP_REPO/$SWIN" ] || die "missing $EP_REPO/$SWIN"

mkdir -p "$DEST/train_run"
MAN=$DEST/FETCH_MANIFEST.txt
echo "# read-only copy from $EP_REPO on $(date '+%Y-%m-%d %H:%M:%S')" > "$MAN"

copy() {
  mkdir -p "$(dirname "$2")"
  cp -- "$1" "$2"
  local a b
  a=$(md5sum < "$1" | cut -d' ' -f1)
  b=$(md5sum < "$2" | cut -d' ' -f1)
  [ "$a" = "$b" ] || die "md5 mismatch after copying $1"
  echo "$a  $1 -> $2" | tee -a "$MAN"
}

copy "$SRC_RUN/$CKPT_NAME" "$DEST/$CKPT_NAME"
copy "$EP_REPO/$SWIN" "$DEST/$(basename "$SWIN")"
for f in "$SRC_RUN"/*.py; do
  if [ -f "$f" ]; then copy "$f" "$DEST/train_run/$(basename "$f")"; fi
done
for d in "$SRC_RUN"/*/; do
  [ -d "$d" ] || continue
  ts=$(basename "$d")
  if [ -f "$d/vis_data/scalars.json" ]; then
    copy "$d/vis_data/scalars.json" "$DEST/train_run/$ts/vis_data/scalars.json"
  fi
  for f in "$d"*.log; do
    if [ -f "$f" ]; then copy "$f" "$DEST/train_run/$ts/$(basename "$f")"; fi
  done
done
ls "$DEST"/train_run/*/vis_data/scalars.json >/dev/null 2>&1 || \
  echo "[fetch] WARNING: no vis_data/scalars.json copied; S0d cannot check BASE values" >&2

if [ -e data/nuscenes ]; then
  echo "[fetch] data/nuscenes already present -> $(readlink -f data/nuscenes)"
else
  tgt=$(readlink -f -- "$EP_REPO/data/nuscenes") || die "cannot resolve $EP_REPO/data/nuscenes"
  [ -f "$tgt/nuscenes_infos_val.pkl" ] || die "no nuscenes_infos_val.pkl under $tgt"
  if under "$tgt" "$EP_REPO"; then
    echo "[fetch] NOTE: nuScenes physically lives inside the EP-Fusion repo;" \
      "the diagnosis will READ it there (read-only, no writes)" >&2
  fi
  mkdir -p data
  ln -s -- "$tgt" data/nuscenes
  echo "[fetch] linked data/nuscenes -> $tgt"
fi
echo "[fetch] done; manifest: $MAN"
