# Confirmed inputs of exec doc #1 (decision record
# projects/DomeDETR/docs/EXEC01_DECISIONS.md, 2026-10-17).
# Source it from the SERVER repo root, e.g.
#   cd /212022085500129/bevfusion-concat/mmdetection3d-claude-jolly-wozniak-c4YMQ
#   source ~/qd_zip/tools/query_diag/exec01_env.sh      (before deployment)
#   source tools/query_diag/exec01_env.sh               (after deployment)
# The R1 threshold (1.0 pp, D-4) is a review-side rule; no script uses it.
_qd_run=work_dirs/bevfusion_lidar-cam_official6e_4xa30_amp512_accum4_seed577127641
if [ ! -d projects/BEVFusion ] || [ ! -d "$_qd_run" ]; then
  echo "[exec01_env] source this from the server repo root" \
    "(missing projects/BEVFusion or $_qd_run)" >&2
  unset _qd_run
  return 1 2>/dev/null || exit 1
fi
export CONFIG=projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_4xa30-amp-accum_nus-3d.py
export CKPT=$PWD/$_qd_run/epoch_5.pth
export TRAIN_WORK_DIR=$PWD/$_qd_run
# Pin the Swin init file to the copy the training run loaded (overwritten by
# CKPT on load; keeps model construction independent of the config's path).
# Non-empty CFG_OPTS must be reported with the results (C3).
export CFG_OPTS="model.img_backbone.init_cfg.checkpoint=$PWD/checkpoints/swint-nuimages-pretrained.pth"
export BASE_MAP=0.6648
export BASE_NDS=0.7060
for _qd_f in "$CONFIG" "$CKPT" "$PWD/checkpoints/swint-nuimages-pretrained.pth"; do
  [ -f "$_qd_f" ] || echo "[exec01_env] WARNING: missing file $_qd_f" >&2
done
echo "[exec01_env] CONFIG=$CONFIG"
echo "[exec01_env] CKPT=$CKPT"
echo "[exec01_env] TRAIN_WORK_DIR=$TRAIN_WORK_DIR"
echo "[exec01_env] CFG_OPTS=$CFG_OPTS"
echo "[exec01_env] BASE_MAP=$BASE_MAP BASE_NDS=$BASE_NDS"
unset _qd_run _qd_f
