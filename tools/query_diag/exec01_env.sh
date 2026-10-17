# Confirmed inputs of exec doc #1 (decision record
# projects/DomeDETR/docs/EXEC01_DECISIONS.md: D-1..D-4, A-1).
# Source it from the root of the Dome-DETR diagnosis workspace (a separate
# checkout of branch claude/dome-detr-query-diag), never from the EP-Fusion
# repo:
#   source tools/query_diag/exec01_env.sh
# Inputs are the local copies made by exec01_fetch_inputs.sh.
# The R1 threshold (1.0 pp, D-4) is a review-side rule; no script uses it.
_qd_ep=${EP_REPO:-/212022085500129/bevfusion-concat/mmdetection3d-claude-jolly-wozniak-c4YMQ}
case "$(readlink -m -- "$PWD")/" in
  "$(readlink -m -- "$_qd_ep")"/*)
    echo "[exec01_env] refusing: cwd is inside the EP-Fusion repo ($_qd_ep)" >&2
    unset _qd_ep
    return 1 2>/dev/null || exit 1 ;;
esac
if [ ! -d projects/BEVFusion ] || [ ! -d tools/query_diag ]; then
  echo "[exec01_env] source this from the Dome-DETR workspace root" >&2
  unset _qd_ep
  return 1 2>/dev/null || exit 1
fi
_qd_in=$PWD/work_dirs/exec01_inputs
export CONFIG=projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_4xa30-amp-accum_nus-3d.py
export CKPT=$_qd_in/epoch_5.pth
export TRAIN_WORK_DIR=$_qd_in/train_run
# Swin init file: local copy of the one the training run loaded. CKPT
# overwrites these weights on load; the pin only keeps model construction
# independent of the path written in the config. Report it (C3).
export CFG_OPTS="model.img_backbone.init_cfg.checkpoint=$_qd_in/swint-nuimages-pretrained.pth"
export BASE_MAP=0.6648
export BASE_NDS=0.7060
for _qd_f in "$CONFIG" "$CKPT" "$_qd_in/swint-nuimages-pretrained.pth" \
             data/nuscenes/nuscenes_infos_val.pkl; do
  [ -e "$_qd_f" ] || echo "[exec01_env] WARNING: missing $_qd_f" \
    "(run tools/query_diag/exec01_fetch_inputs.sh first)" >&2
done
echo "[exec01_env] CONFIG=$CONFIG"
echo "[exec01_env] CKPT=$CKPT"
echo "[exec01_env] TRAIN_WORK_DIR=$TRAIN_WORK_DIR"
echo "[exec01_env] CFG_OPTS=$CFG_OPTS"
echo "[exec01_env] BASE_MAP=$BASE_MAP BASE_NDS=$BASE_NDS"
unset _qd_ep _qd_in _qd_f
