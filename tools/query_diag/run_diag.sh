#!/usr/bin/env bash
# Query-initialization diagnostics for BEVFusion (exec doc #1 v1.1).
# Run from the mmdetection3d repo root. The script itself may live outside
# the repo (e.g. in an unpacked branch zip): that is how S0 is run before
# tools/query_diag/ is copied into the server repo.
#
#   CONFIG=.. CKPT=.. [TRAIN_WORK_DIR=work_dirs/baseline1] run_diag.sh s0
#                                          S0a-S0d consistency checks, CPU only
#   CONFIG=.. CKPT=.. CUDA_VISIBLE_DEVICES=<id> run_diag.sh quick
#                                          300 frames (step 1 smoke test)
#   CONFIG=.. CKPT=.. CUDA_VISIBLE_DEVICES=<id> run_diag.sh recall
#                                          full val, D1 + D3 (step 2)
#   run_diag.sh merge                      merged bins, rule inputs, S1-S4a
#   CONFIG=.. CKPT=.. BASE_MAP=.. BASE_NDS=.. GPUS=1 CUDA_VISIBLE_DEVICES=<id> \
#     run_diag.sh oracle                   base -> S5 gate -> 4 oracle groups
#   CONFIG=.. CKPT=.. CUDA_VISIBLE_DEVICES=<id> run_diag.sh s5-adjudicate
#                                          only after an S5 FAIL (sec. 8)
#   run_diag.sh summary
#
# Env: CONFIG, CKPT    required (no defaults)
#      CFG_OPTS        optional, passed to --cfg-options; recorded in every
#                      output json (report it whenever it is non-empty)
#      BASE_MAP/BASE_NDS  required by 'oracle' (4 decimals, same eval setup)
#      GPUS            oracle only, 1 (recommended) or 2
#      TRAIN_WORK_DIR  s0 only, optional: training work dir of the checkpoint
#                      (dumped config + val logs) for S0d
#      PORT, OUT       defaults 29531, work_dirs/diag
#
# Every GPU step first checks each GPU in CUDA_VISIBLE_DEVICES (C5,
# mandatory, no override): it refuses to start if any compute process is
# listed or memory.used > 1024 MiB (processes of other containers may be
# invisible to nvidia-smi).
set -euo pipefail

DIAG=$(cd "$(dirname "$0")" && pwd)
GPUS=${GPUS:-1}
PORT=${PORT:-29531}
OUT=${OUT:-work_dirs/diag}
STEP=${1:-}
MEM_LIMIT_MIB=1024
export PYTHONPATH=$PWD:$DIAG:${PYTHONPATH:-}
# nvidia-smi enumerates GPUs in PCI bus order; make CUDA use the same order
# so that the ids checked below are the ids the job actually runs on.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

die() { echo "[run_diag] $*" >&2; exit 2; }

CFG_ARGS=()
if [ -n "${CFG_OPTS:-}" ]; then
  read -r -a _opts <<< "$CFG_OPTS"
  CFG_ARGS=(--cfg-options "${_opts[@]}")
  echo "[run_diag] NOTE: CFG_OPTS in effect: $CFG_OPTS (recorded in json)"
fi

need_repo_root() {
  [ -d projects/BEVFusion ] && [ -f tools/test.py ] || \
    die "run from the mmdetection3d repo root (cwd: $PWD)"
}
need_config() {
  [ -n "${CONFIG:-}" ] || die "set CONFIG=<config path> (required, no default)"
  [ -f "$CONFIG" ] || die "CONFIG not found: $CONFIG"
}
need_ckpt() {
  [ -n "${CKPT:-}" ] || die "set CKPT=<checkpoint path> (required)"
  [ -f "$CKPT" ] || die "CKPT not found: $CKPT"
}
need_base() {
  [ -n "${BASE_MAP:-}" ] && [ -n "${BASE_NDS:-}" ] || \
    die "set BASE_MAP and BASE_NDS (4 decimals) before 'oracle'"
}

# C5: mandatory idle check of every GPU listed in CUDA_VISIBLE_DEVICES.
check_gpus() {
  local want=$1 id apps mem
  local -a ids
  command -v nvidia-smi >/dev/null 2>&1 || \
    die "[gpu-check] nvidia-smi not found; refusing to start"
  [ -n "${CUDA_VISIBLE_DEVICES:-}" ] || \
    die "[gpu-check] set CUDA_VISIBLE_DEVICES to the idle GPU id(s)"
  IFS=',' read -r -a ids <<< "$CUDA_VISIBLE_DEVICES"
  [ "${#ids[@]}" -eq "$want" ] || die "[gpu-check] this step needs exactly" \
    "$want GPU(s); CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  for id in "${ids[@]}"; do
    id=${id//[[:space:]]/}
    apps=$(nvidia-smi -i "$id" --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader 2>&1) || die "[gpu-check] nvidia-smi failed for GPU $id: $apps"
    apps=$(printf '%s\n' "$apps" | grep -v -i 'no running processes' | \
      sed '/^[[:space:]]*$/d' || true)
    if [ -n "$apps" ]; then
      echo "[gpu-check] GPU $id has compute processes; refusing to start:" >&2
      echo "$apps" >&2
      exit 3
    fi
    mem=$(nvidia-smi -i "$id" --query-gpu=memory.used \
      --format=csv,noheader,nounits 2>&1) || die "[gpu-check] nvidia-smi failed for GPU $id: $mem"
    mem=${mem//[[:space:]]/}
    [[ "$mem" =~ ^[0-9]+$ ]] || die "[gpu-check] cannot parse memory.used for GPU $id: '$mem'"
    if [ "$mem" -gt "$MEM_LIMIT_MIB" ]; then
      echo "[gpu-check] GPU $id memory.used=${mem} MiB > ${MEM_LIMIT_MIB} MiB" \
        "(possibly another container); refusing to start" >&2
      exit 3
    fi
    echo "[gpu-check] GPU $id idle: no compute process, memory.used=${mem} MiB"
  done
}

run_oracle() {
  local tag=$1; shift
  check_gpus "$GPUS"
  if [ "$GPUS" -gt 1 ]; then
    torchrun --nproc_per_node="$GPUS" --master_port="$PORT" \
      "$DIAG/diag_oracle_query.py" "$CONFIG" "$CKPT" --launcher pytorch \
      --work-dir "$OUT/oracle_$tag" --tag "$tag" "$@" \
      ${CFG_ARGS[@]+"${CFG_ARGS[@]}"}
  else
    python "$DIAG/diag_oracle_query.py" "$CONFIG" "$CKPT" \
      --work-dir "$OUT/oracle_$tag" --tag "$tag" "$@" \
      ${CFG_ARGS[@]+"${CFG_ARGS[@]}"}
  fi
}

case "$STEP" in
  s0)
    need_repo_root; need_config; need_ckpt
    S0_ARGS=(--config "$CONFIG" --ckpt "$CKPT" --out-dir "$OUT/s0")
    if [ -n "${TRAIN_WORK_DIR:-}" ]; then
      S0_ARGS+=(--train-work-dir "$TRAIN_WORK_DIR")
    fi
    # CPU only: the consistency checker never touches a GPU.
    CUDA_VISIBLE_DEVICES="" python "$DIAG/diag_s0_check.py" "${S0_ARGS[@]}" \
      ${CFG_ARGS[@]+"${CFG_ARGS[@]}"} ;;
  quick)
    need_repo_root; need_config; need_ckpt; check_gpus 1
    python "$DIAG/diag_query_recall.py" "$CONFIG" "$CKPT" \
      --out-dir "$OUT/recall_quick" --max-samples 300 \
      ${CFG_ARGS[@]+"${CFG_ARGS[@]}"} ;;
  recall)
    need_repo_root; need_config; need_ckpt; check_gpus 1
    python "$DIAG/diag_query_recall.py" "$CONFIG" "$CKPT" \
      --out-dir "$OUT/recall" ${CFG_ARGS[@]+"${CFG_ARGS[@]}"} ;;
  merge)
    MERGE_DIRS=()
    for d in "$OUT/recall_quick" "$OUT/recall"; do
      if [ -f "$d/recall_per_gt.npz" ]; then MERGE_DIRS+=("$d"); fi
    done
    [ "${#MERGE_DIRS[@]}" -gt 0 ] || die "no recall_per_gt.npz under $OUT/recall_quick or $OUT/recall"
    python "$DIAG/diag_merge_bins.py" "${MERGE_DIRS[@]}" ;;
  oracle)
    need_repo_root; need_config; need_ckpt; need_base
    [ "$GPUS" = 1 ] || [ "$GPUS" = 2 ] || die "GPUS must be 1 or 2 (got $GPUS)"
    run_oracle base --oracle none
    if ! python "$DIAG/diag_gate_s5.py" gate \
        --base-json "$OUT/oracle_base/oracle_base.json" \
        --map "$BASE_MAP" --nds "$BASE_NDS" --config "$CONFIG" --ckpt "$CKPT" \
        --out "$OUT/oracle_base/s5_gate.json"; then
      echo "[run_diag] S5 FAILED: the remaining four oracle groups were NOT run." >&2
      echo "[run_diag] Next (pre-registered): run_diag.sh s5-adjudicate, then report." >&2
      exit 4
    fi
    run_oracle all     --oracle loc     --range 0 1000
    run_oracle far     --oracle loc     --range 30 1000
    run_oracle near    --oracle loc     --range 0 30
    run_oracle all_cls --oracle loc_cls --range 0 1000 ;;
  s5-adjudicate)
    need_repo_root; need_config; need_ckpt
    [ -f "$OUT/oracle_base/oracle_base.json" ] || \
      die "no $OUT/oracle_base/oracle_base.json; run 'oracle' first"
    check_gpus 1
    # Un-hooked official entry point, same CONFIG / CKPT / CFG_OPTS, 1 GPU.
    python tools/test.py "$CONFIG" "$CKPT" --work-dir "$OUT/s5_nohook" \
      ${CFG_ARGS[@]+"${CFG_ARGS[@]}"}
    ADJ_ARGS=(--base-json "$OUT/oracle_base/oracle_base.json"
              --nohook-dir "$OUT/s5_nohook" --config "$CONFIG" --ckpt "$CKPT"
              --out "$OUT/s5_nohook/s5_adjudication.json")
    if [ -n "${BASE_MAP:-}" ] && [ -n "${BASE_NDS:-}" ]; then
      ADJ_ARGS+=(--map "$BASE_MAP" --nds "$BASE_NDS")
    fi
    if python "$DIAG/diag_gate_s5.py" adjudicate "${ADJ_ARGS[@]}"; then
      echo "[run_diag] verdict SOURCE. STOP: report both results and wait for" \
        "confirmation before using any new reference value." >&2
    else
      echo "[run_diag] verdict HOOK. STOP and report." >&2
    fi
    exit 4 ;;
  summary)
    python "$DIAG/diag_summarize.py" "$OUT" ;;
  *)
    echo "unknown step: '$STEP' (s0|quick|recall|merge|oracle|s5-adjudicate|summary)"
    exit 1 ;;
esac
