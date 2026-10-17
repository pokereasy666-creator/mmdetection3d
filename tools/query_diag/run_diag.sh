#!/usr/bin/env bash
# Query-initialization diagnostics for BEVFusion (mmdet3d 1.4, projects/BEVFusion)
# Run from the mmdetection3d repo root.
#
#   CKPT=/path/to/bevfusion.pth bash /path/to/diag/run_diag.sh quick    # 300 frames, ~5 min
#   CKPT=/path/to/bevfusion.pth bash /path/to/diag/run_diag.sh recall   # full val, diag 1+3
#   CKPT=/path/to/bevfusion.pth GPUS=2 bash /path/to/diag/run_diag.sh oracle  # diag 2, 5 runs
#   bash /path/to/diag/run_diag.sh summary
#
# Env: CONFIG (default: BEVFusion lidar-cam config), CKPT (required except summary),
#      GPUS (default 1, oracle only), PORT, OUT (default work_dirs/diag)
set -euo pipefail

DIAG=$(cd "$(dirname "$0")" && pwd)
CONFIG=${CONFIG:-projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py}
GPUS=${GPUS:-1}
PORT=${PORT:-29531}
OUT=${OUT:-work_dirs/diag}
STEP=${1:-quick}
export PYTHONPATH=$PWD:$DIAG:${PYTHONPATH:-}

need_ckpt() { : "${CKPT:?set CKPT=/path/to/checkpoint.pth}"; }

run_oracle() {
  local tag=$1; shift
  if [ "$GPUS" -gt 1 ]; then
    torchrun --nproc_per_node="$GPUS" --master_port="$PORT" \
      "$DIAG/diag_oracle_query.py" "$CONFIG" "$CKPT" --launcher pytorch \
      --work-dir "$OUT/oracle_$tag" --tag "$tag" "$@"
  else
    python "$DIAG/diag_oracle_query.py" "$CONFIG" "$CKPT" \
      --work-dir "$OUT/oracle_$tag" --tag "$tag" "$@"
  fi
}

case "$STEP" in
  quick)
    need_ckpt
    python "$DIAG/diag_query_recall.py" "$CONFIG" "$CKPT" \
      --out-dir "$OUT/recall_quick" --max-samples 300 ;;
  recall)
    need_ckpt
    python "$DIAG/diag_query_recall.py" "$CONFIG" "$CKPT" \
      --out-dir "$OUT/recall" ;;
  oracle)
    need_ckpt
    run_oracle base    --oracle none
    run_oracle all     --oracle loc     --range 0 1000
    run_oracle far     --oracle loc     --range 30 1000
    run_oracle near    --oracle loc     --range 0 30
    run_oracle all_cls --oracle loc_cls --range 0 1000 ;;
  summary)
    python "$DIAG/diag_summarize.py" "$OUT" ;;
  *)
    echo "unknown step: $STEP (quick|recall|oracle|summary)"; exit 1 ;;
esac
