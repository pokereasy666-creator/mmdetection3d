#!/usr/bin/env bash
# 等待共享节点上出现 N 张空闲 GPU，连续确认后自动启动 BEVFusion DGF 训练。
set -uo pipefail

# ===================== 可调参数 =====================
NEED_GPUS=${NEED_GPUS:-2}                       # 需要的空闲 GPU 数量
MEM_FREE_THRESHOLD_MB=${MEM_FREE_THRESHOLD_MB:-1000}  # 显存占用低于该值(MiB)视为空闲
POLL_INTERVAL=${POLL_INTERVAL:-15}              # 轮询间隔(秒)
STABLE_COUNT=${STABLE_COUNT:-3}                 # 需连续多少次确认空闲才启动(防抖/防抢)
# 项目根目录：默认 = 本脚本所在目录的上一级(即 tools/ 的父目录)
PROJECT_DIR=${PROJECT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
WORK_DIR=${WORK_DIR:-work_dirs/dgf_2gpu_ls512}
LOG_FILE=${LOG_FILE:-dgf_2gpu_ls512.log}
# ====================================================

log() { echo "[$(date '+%F %T')] $*"; }

# 列出显存占用低于阈值的 GPU index（每行一个）
get_free_gpus() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' -v th="$MEM_FREE_THRESHOLD_MB" '{ if (($2+0) < th) print ($1+0) }'
}

command -v nvidia-smi >/dev/null 2>&1 || { log "找不到 nvidia-smi，退出"; exit 1; }

log "开始监控 GPU：需要 ${NEED_GPUS} 张空闲(显存<${MEM_FREE_THRESHOLD_MB}MiB)，"\
"连续 ${STABLE_COUNT} 次确认后启动，轮询间隔 ${POLL_INTERVAL}s"
log "项目目录：${PROJECT_DIR}"

stable=0
free=()
while true; do
  mapfile -t free < <(get_free_gpus)
  n=${#free[@]}
  if [ "$n" -ge "$NEED_GPUS" ]; then
    stable=$((stable + 1))
    log "检测到 ${n} 张空闲 GPU: [${free[*]}]（连续确认 ${stable}/${STABLE_COUNT}）"
    [ "$stable" -ge "$STABLE_COUNT" ] && break
  else
    [ "$stable" -ne 0 ] && log "空闲卡数变少，连续确认计数清零"
    stable=0
    log "当前空闲 GPU 数 ${n} < ${NEED_GPUS}，继续等待…"
  fi
  sleep "$POLL_INTERVAL"
done

# 取前 NEED_GPUS 张空闲卡
sel=("${free[@]:0:NEED_GPUS}")
GPU_LIST=$(IFS=,; echo "${sel[*]}")
PORT=$(( 20000 + RANDOM % 20000 ))   # 唯一端口，避免和别人的 DDP 撞 29500

log "==> 启动训练：CUDA_VISIBLE_DEVICES=${GPU_LIST}  PORT=${PORT}"

cd "$PROJECT_DIR" || { log "无法进入 ${PROJECT_DIR}"; exit 1; }

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
