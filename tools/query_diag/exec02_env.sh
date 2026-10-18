# Inputs of exec doc #2 (projects/DomeDETR/docs/
# exec_doc_02_coverage_vs_localization.md): the same CONFIG / CKPT /
# CFG_OPTS as exec doc #1 (taken from exec01_env.sh), and BASE_MAP /
# BASE_NDS set to the exec doc #2 references (= exec doc #1 step 3 base run,
# used by S5'). Source it from the Dome-DETR workspace root:
#   source tools/query_diag/exec02_env.sh
_qd_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if ! . "$_qd_dir/exec01_env.sh"; then
  unset _qd_dir
  return 1 2>/dev/null || exit 1
fi
export BASE_MAP=0.665463
export BASE_NDS=0.706154
echo "[exec02_env] BASE_MAP=$BASE_MAP BASE_NDS=$BASE_NDS" \
  "(exec doc #2 references: exec doc #1 step 3 base; overrides the line above)"
unset _qd_dir
