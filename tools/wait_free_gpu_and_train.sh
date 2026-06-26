#!/usr/bin/env bash
# Wait until N GPUs are free on a shared node, then auto-launch BEVFusion DGF
# training after several consecutive "free" confirmations.
set -uo pipefail

# ===================== tunables =====================
NEED_GPUS=${NEED_GPUS:-2}                       # number of free GPUs required
MEM_FREE_THRESHOLD_MB=${MEM_FREE_THRESHOLD_MB:-1000}  # a GPU is "free" if used mem (MiB) is below this
POLL_INTERVAL=${POLL_INTERVAL:-15}              # polling interval (seconds)
STABLE_COUNT=${STABLE_COUNT:-3}                 # consecutive free confirmations before launching (debounce)
# project root: defaults to the parent of this script's dir (i.e. parent of tools/)
PROJECT_DIR=${PROJECT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
WORK_DIR=${WORK_DIR:-work_dirs/dgf_2gpu_ls512}
LOG_FILE=${LOG_FILE:-dgf_2gpu_ls512.log}
# ====================================================

log() { echo "[$(date '+%F %T')] $*"; }

# print the index of each GPU whose used memory is below the threshold (one per line)
get_free_gpus() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' -v th="$MEM_FREE_THRESHOLD_MB" '{ if (($2+0) < th) print ($1+0) }'
}

command -v nvidia-smi >/dev/null 2>&1 || { log "nvidia-smi not found, exiting"; exit 1; }

log "Monitoring GPUs: need ${NEED_GPUS} free (used mem < ${MEM_FREE_THRESHOLD_MB}MiB), "\
"launch after ${STABLE_COUNT} consecutive confirmations, poll every ${POLL_INTERVAL}s"
log "Project dir: ${PROJECT_DIR}"

stable=0
free=()
while true; do
  mapfile -t free < <(get_free_gpus)
  n=${#free[@]}
  if [ "$n" -ge "$NEED_GPUS" ]; then
    stable=$((stable + 1))
    log "Found ${n} free GPU(s): [${free[*]}] (confirmed ${stable}/${STABLE_COUNT})"
    [ "$stable" -ge "$STABLE_COUNT" ] && break
  else
    [ "$stable" -ne 0 ] && log "Fewer free GPUs than before, resetting confirmation counter"
    stable=0
    log "Free GPUs ${n} < ${NEED_GPUS}, keep waiting..."
  fi
  sleep "$POLL_INTERVAL"
done

# take the first NEED_GPUS free GPUs
sel=("${free[@]:0:NEED_GPUS}")
GPU_LIST=$(IFS=,; echo "${sel[*]}")
PORT=$(( 20000 + RANDOM % 20000 ))   # unique port to avoid clashing with others' DDP on 29500

log "==> Launching training: CUDA_VISIBLE_DEVICES=${GPU_LIST}  PORT=${PORT}"

cd "$PROJECT_DIR" || { log "Cannot cd into ${PROJECT_DIR}"; exit 1; }

DGF_DEBUG=1 DGF_DEBUG_EVERY=50 \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_LIST" PORT="$PORT" \
bash tools/dist_train.sh \
  projects/BEVFusion/configs/bevfusion_lidar-cam_4xa30_dgf_nus-3d.py "$NEED_GPUS" \
  --sync_bn torch \
  --work-dir "$WORK_DIR" \
  --cfg-options train_cfg.max_epochs=1 \
    randomness.seed=577127641 \
    optim_wrapper.loss_scale=512.0 \
    model.img_backbone.init_cfg.checkpoint="$PROJECT_DIR/checkpoints/swint-nuimages-pretrained.pth" \
    load_from="$PROJECT_DIR/checkpoints/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth" \
  2>&1 | tee "$LOG_FILE"
