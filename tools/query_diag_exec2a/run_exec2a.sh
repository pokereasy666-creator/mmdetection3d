#!/usr/bin/env bash
# Exec doc #2a (E3'): two oracle runs with the UNCHANGED
# tools/query_diag/diag_oracle_query.py and existing flags only, plus S0'' and
# the summary. Run from the Dome-DETR workspace root after
#   source tools/query_diag/exec02_env.sh
#
#   run_exec2a.sh s0pp                     S0'' only (no GPU)
#   CUDA_VISIBLE_DEVICES=<id> run_exec2a.sh run
#                                          S0'' -> E3'-exact -> its S6/S7'/S8'
#                                          -> E3'-jitter -> summary (1 GPU)
#   run_exec2a.sh summary                  re-run the summary only
#
# Outputs: $OUT/exec2a/ (oracle_e3p_exact/, oracle_e3p_jitter/,
# s0pp_check.*, exec2a_summary.*) and $OUT/logs/exec2a-<step>_<time>.log.
# Reads: $OUT/exec2/oracle_*/oracle_*.json (exec doc #2 step 2; E0 is the
# reference for every delta) and $OUT/exec2/b0_decoder.json (S8' size).
# The guards below (A-1 protected dirs, C5 GPU check, CFG_OPTS) are copied
# verbatim from tools/query_diag/run_diag.sh, which must stay unchanged.
set -euo pipefail

X2A=$(cd "$(dirname "$0")" && pwd)
DIAG=$(cd "$X2A/../query_diag" && pwd)
OUT=${OUT:-work_dirs/diag}
ROOT=$OUT/exec2a
STEP=${1:-}
MEM_LIMIT_MIB=1024
export PYTHONPATH=$PWD:$DIAG:${PYTHONPATH:-}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
PROTECTED_DIRS=${PROTECTED_DIRS:-/212022085500129/bevfusion-concat/mmdetection3d-claude-jolly-wozniak-c4YMQ}

die() { echo "[exec2a] $*" >&2; exit 2; }

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
  s0pp|run|summary)
    mkdir -p "$OUT/logs"
    RUN_LOG="$OUT/logs/exec2a-${STEP}_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "$RUN_LOG") 2>&1
    echo "[exec2a] step=$STEP log=$RUN_LOG" ;;
  *)
    echo "usage: run_exec2a.sh {s0pp|run|summary} (got '$STEP')"
    exit 1 ;;
esac

CFG_ARGS=()
if [ -n "${CFG_OPTS:-}" ]; then
  read -r -a _opts <<< "$CFG_OPTS"
  CFG_ARGS=(--cfg-options "${_opts[@]}")
  echo "[exec2a] NOTE: CFG_OPTS in effect: $CFG_OPTS (recorded in json)"
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
need_exec2() {
  [ -f "$OUT/exec2/oracle_e0_base/oracle_e0_base.json" ] || \
    die "exec doc #2 E0 $OUT/exec2/oracle_e0_base/oracle_e0_base.json not found"
  [ -f "$OUT/exec2/b0_decoder.json" ] || \
    die "exec doc #2 B0 $OUT/exec2/b0_decoder.json not found"
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

s0pp() {
  if ! python "$X2A/exec2a_check.py" s0pp --qd "$DIAG" \
      --ref-dir "$OUT/exec2" --config "$CONFIG" --ckpt "$CKPT" \
      --out-dir "$ROOT"; then
    echo "[exec2a] S0'' FAILED: no oracle run started. Stop and report." >&2
    exit 4
  fi
}

run_e3p() {
  local tag=$1; shift
  check_gpus 1                              # exec doc #2a: one GPU only
  python "$DIAG/diag_oracle_query.py" "$CONFIG" "$CKPT" \
    --work-dir "$ROOT/oracle_$tag" --tag "$tag" "$@" \
    ${CFG_ARGS[@]+"${CFG_ARGS[@]}"}
}

case "$STEP" in
  s0pp)
    need_repo_root; need_config; need_ckpt; need_exec2
    s0pp ;;
  run)
    need_repo_root; need_config; need_ckpt; need_exec2
    s0pp
    # exec doc #2a sec. 2, verbatim
    run_e3p e3p_exact --oracle loc --subset missed --miss-thr 0.5 \
      --range 0 1000 --score orig
    if ! python "$X2A/exec2a_check.py" check-run \
        --json "$ROOT/oracle_e3p_exact/oracle_e3p_exact.json" \
        --b0 "$OUT/exec2/b0_decoder.json"; then
      echo "[exec2a] E3'-exact failed a self-check: E3'-jitter was NOT run." \
        "Stop and report." >&2
      python "$X2A/exec2a_check.py" summary --root "$ROOT" --qd "$DIAG" \
        --e0 "$OUT/exec2/oracle_e0_base/oracle_e0_base.json" \
        --b0 "$OUT/exec2/b0_decoder.json" || true
      exit 4
    fi
    run_e3p e3p_jitter --oracle loc --subset missed --miss-thr 0.5 \
      --range 0 1000 --score orig --jitter-cells 1 --seed 0
    if ! python "$X2A/exec2a_check.py" summary --root "$ROOT" --qd "$DIAG" \
        --e0 "$OUT/exec2/oracle_e0_base/oracle_e0_base.json" \
        --b0 "$OUT/exec2/b0_decoder.json"; then
      echo "[exec2a] a self-check FAILED (see the summary). Stop and report." >&2
      exit 4
    fi ;;
  summary)
    python "$X2A/exec2a_check.py" summary --root "$ROOT" --qd "$DIAG" \
      --e0 "$OUT/exec2/oracle_e0_base/oracle_e0_base.json" \
      --b0 "$OUT/exec2/b0_decoder.json" ;;
esac
