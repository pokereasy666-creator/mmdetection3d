#!/usr/bin/env bash
# Exec doc #3 (QueryLoc) runner. Run from the Dome-DETR workspace root after
#   source tools/query_diag/exec02_env.sh    (CONFIG / CKPT / CFG_OPTS of #1/#2)
#
#   run_queryloc.sh probe                     read-only server probe (no GPU)
#   run_queryloc.sh s0ppp                     S0''' only (no GPU)
#   CUDA_VISIBLE_DEVICES=<id> run_queryloc.sh 0a
#                                             S0''' -> C5 -> step 0a (1 GPU)
# The steps of 4.2 / 4.3 and of steps 1-2 are added after the plan is
# confirmed.
#
# Outputs under $OUT (default work_dirs/queryloc): probe_step0_<time>.txt,
# s0ppp/, step0a/, logs/<step>_<time>.log. Reference: $REF_E0 (exec doc #2
# E0 json, default work_dirs/diag/exec2/oracle_e0_base/oracle_e0_base.json).
# The A-1 guard, C5 GPU check and CFG_OPTS handling are copied verbatim from
# tools/query_diag/run_diag.sh (which must stay unchanged).
set -euo pipefail

QL=$(cd "$(dirname "$0")" && pwd)
DIAG=$(cd "$QL/../../../tools/query_diag" && pwd)
OUT=${OUT:-work_dirs/queryloc}
REF_E0=${REF_E0:-work_dirs/diag/exec2/oracle_e0_base/oracle_e0_base.json}
STEP=${1:-}
MEM_LIMIT_MIB=1024
export PYTHONPATH=$PWD:$DIAG:${PYTHONPATH:-}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
PROTECTED_DIRS=${PROTECTED_DIRS:-/212022085500129/bevfusion-concat/mmdetection3d-claude-jolly-wozniak-c4YMQ}

die() { echo "[queryloc] $*" >&2; exit 2; }

inside_protected() {
  local p d
  p=$(readlink -m -- "$1")
  for d in $PROTECTED_DIRS; do
    d=$(readlink -m -- "$d")
    case "$p/" in "$d"/*) return 0 ;; esac
  done
  return 1
}
if inside_protected "$PWD"; then
  die "cwd $PWD is inside a protected directory ($PROTECTED_DIRS); run from" \
    "the separate Dome-DETR workspace"
fi
if inside_protected "$OUT"; then
  die "OUT=$OUT resolves inside a protected directory ($PROTECTED_DIRS)"
fi
case "$STEP" in
  probe|s0ppp|0a)
    mkdir -p "$OUT/logs"
    RUN_LOG="$OUT/logs/${STEP}_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "$RUN_LOG") 2>&1
    echo "[queryloc] step=$STEP log=$RUN_LOG" ;;
  *)
    echo "usage: run_queryloc.sh {probe|s0ppp|0a} (got '$STEP')"
    exit 1 ;;
esac

CFG_ARGS=()
if [ -n "${CFG_OPTS:-}" ]; then
  read -r -a _opts <<< "$CFG_OPTS"
  CFG_ARGS=(--cfg-options "${_opts[@]}")
  echo "[queryloc] NOTE: CFG_OPTS in effect: $CFG_OPTS (recorded in json)"
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
need_ref() {
  [ -f "$REF_E0" ] || die "exec doc #2 E0 json $REF_E0 not found"
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

s0ppp() {
  if ! python "$QL/queryloc_check.py" s0ppp --config "$CONFIG" \
      --ckpt "$CKPT" --ref-json "$REF_E0" --out-dir "$OUT/s0ppp"; then
    echo "[queryloc] S0''' FAILED: nothing was started. Stop and report." >&2
    exit 4
  fi
}

case "$STEP" in
  probe)
    python "$QL/probe_step0.py" --out-dir "$OUT" ;;
  s0ppp)
    need_repo_root; need_config; need_ckpt; need_ref
    s0ppp ;;
  0a)
    need_repo_root; need_config; need_ckpt; need_ref
    s0ppp
    check_gpus 1
    if ! python "$QL/subcell_0a.py" "$CONFIG" "$CKPT" \
        --out-dir "$OUT/step0a" ${CFG_ARGS[@]+"${CFG_ARGS[@]}"}; then
      echo "[queryloc] step 0a: a self-check FAILED (see subcell_0a.txt)." \
        "Stop and report." >&2
      exit 4
    fi ;;
esac
