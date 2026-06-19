#!/usr/bin/env bash
# =============================================================================
# deploy.sh — EP-Fusion 离线服务器部署脚本（zip-by-SHA 工作流）
#
# 用途：在"GitHub 下载 zip → 上传离线服务器 → 解压为全新 SHA 目录"之后，
#       把该目录变成可运行状态。服务器无 git / 无网络 / 不能 pip install
#       （CLAUDE.md 铁律 15/16/17）。
#
# 动作（幂等，可重复执行）：
#   1) 打印 VERSION（无 git，铁律 16）；
#   2) 建软链 data/nuscenes 与 work_dirs（指向真实数据/训练产物目录）；
#   3) 从旧部署目录拷贝 BEVFusion 自定义 CUDA 算子编译产物 *.so
#      （setup.py 为原地编译，zip 不含 .so，缺了连 import 都过不去）；
#   4) 环境自检：import torch/mmengine/mmcv/mmdet/mmdet3d 打印版本与 __file__，
#      并验证 ops 可 import。
#
# 用法（在解压出的仓库根目录执行）：
#   bash scripts/deploy.sh \
#       --data      /path/to/nuscenes \
#       --work-dirs /path/to/work_dirs \
#       --ops-from  /path/to/旧部署或训练仓库根目录
# 可选：--skip-ops（已手动处理 ops 时）  --python <bin>（默认 python）
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_PATH=""
WORK_DIRS=""
OPS_FROM=""
SKIP_OPS=0
PYTHON_BIN="${PYTHON:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data)      DATA_PATH="$2"; shift 2 ;;
    --work-dirs) WORK_DIRS="$2"; shift 2 ;;
    --ops-from)  OPS_FROM="$2"; shift 2 ;;
    --skip-ops)  SKIP_OPS=1; shift ;;
    --python)    PYTHON_BIN="$2"; shift 2 ;;
    -h|--help)   grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "[deploy] 未知参数: $1（--help 查看用法）" >&2; exit 1 ;;
  esac
done

echo "============================================================"
echo "[deploy] 仓库根目录: ${REPO_ROOT}"
if [[ -f "${REPO_ROOT}/VERSION" ]]; then
  echo "[deploy] VERSION: $(cat "${REPO_ROOT}/VERSION" | head -1)"
else
  echo "[deploy] VERSION: unknown（VERSION 文件缺失——铁律 16 要求每次 commit 更新）"
fi
echo "============================================================"

# ---------- 2) 数据与 work_dirs 软链 ----------
if [[ -n "${DATA_PATH}" ]]; then
  mkdir -p "${REPO_ROOT}/data"
  if [[ -e "${REPO_ROOT}/data/nuscenes" && ! -L "${REPO_ROOT}/data/nuscenes" ]]; then
    echo "[deploy] data/nuscenes 已存在且不是软链，拒绝覆盖（请人工确认）" >&2; exit 1
  fi
  ln -sfn "${DATA_PATH}" "${REPO_ROOT}/data/nuscenes"
  echo "[deploy] 软链 data/nuscenes -> ${DATA_PATH}"
else
  if [[ -e "${REPO_ROOT}/data/nuscenes" ]]; then
    echo "[deploy] data/nuscenes 已就绪（未指定 --data，保持现状）"
  else
    echo "[deploy] 警告：未指定 --data 且 data/nuscenes 不存在；探针 P3-P5 与训练将不可用"
  fi
fi

if [[ -n "${WORK_DIRS}" ]]; then
  if [[ -e "${REPO_ROOT}/work_dirs" && ! -L "${REPO_ROOT}/work_dirs" ]]; then
    echo "[deploy] work_dirs 已存在且不是软链，拒绝覆盖（请人工确认）" >&2; exit 1
  fi
  ln -sfn "${WORK_DIRS}" "${REPO_ROOT}/work_dirs"
  echo "[deploy] 软链 work_dirs -> ${WORK_DIRS}"
fi

# ---------- 3) BEVFusion 自定义 CUDA 算子 ----------
# projects/BEVFusion/setup.py 用 CUDAExtension 原地编译（develop / build_ext --inplace），
# .so 落在源码树 projects/BEVFusion/bevfusion/ops/{voxel,bev_pool}/ 内；
# 全新解压目录不含 .so，而 projects.BEVFusion.bevfusion/__init__.py 链式 import
# 会立即执行 `from .ops import Voxelization` —— 缺 .so 时 import 即失败。
OPS_DIR="${REPO_ROOT}/projects/BEVFusion/bevfusion/ops"
count_so() { find "${OPS_DIR}" -name '*.so' 2>/dev/null | wc -l; }

if [[ "${SKIP_OPS}" -eq 1 ]]; then
  echo "[deploy] --skip-ops：跳过算子处理（当前 .so 数量: $(count_so)）"
elif [[ "$(count_so)" -gt 0 ]]; then
  echo "[deploy] 检测到已有编译产物（.so 数量: $(count_so)），跳过拷贝"
elif [[ -n "${OPS_FROM}" ]]; then
  SRC_OPS="${OPS_FROM}/projects/BEVFusion/bevfusion/ops"
  if [[ ! -d "${SRC_OPS}" ]]; then
    echo "[deploy] 错误：${SRC_OPS} 不存在（--ops-from 应指向旧部署/训练仓库根目录）" >&2; exit 1
  fi
  N=0
  while IFS= read -r so; do
    rel="${so#"${SRC_OPS}"/}"
    mkdir -p "${OPS_DIR}/$(dirname "${rel}")"
    cp -p "${so}" "${OPS_DIR}/${rel}"
    echo "[deploy]   拷贝 ${rel}"
    N=$((N+1))
  done < <(find "${SRC_OPS}" -name '*.so')
  if [[ "${N}" -eq 0 ]]; then
    echo "[deploy] 错误：${SRC_OPS} 下未找到任何 .so——旧目录可能也未编译。" >&2
    echo "[deploy] 重编译命令（需 nvcc，与训练同 conda 环境）：" >&2
    echo "[deploy]   cd ${REPO_ROOT} && ${PYTHON_BIN} projects/BEVFusion/setup.py build_ext --inplace" >&2
    exit 1
  fi
  echo "[deploy] 算子拷贝完成（${N} 个 .so；同机同环境同 GPU 架构可直接复用）"
else
  echo "[deploy] 错误：缺少编译产物且未指定 --ops-from。" >&2
  echo "[deploy] 二选一：" >&2
  echo "[deploy]   a) bash scripts/deploy.sh --ops-from /path/to/旧部署目录 ..." >&2
  echo "[deploy]   b) 重编译: cd ${REPO_ROOT} && ${PYTHON_BIN} projects/BEVFusion/setup.py build_ext --inplace" >&2
  exit 1
fi

# ---------- 4) 环境自检 ----------
echo "[deploy] 环境自检（PYTHONPATH=${REPO_ROOT}）..."
# 临时关闭 errexit：自检 python 返回非零时要走到下方 RC 判定打印友好提示，
# 而不是被 set -e 在此处直接中断。
set +e
PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" - <<'PYEOF'
import importlib, sys
ok = True
for name in ('torch', 'mmengine', 'mmcv', 'mmdet', 'mmdet3d'):
    try:
        m = importlib.import_module(name)
        print(f"[selfcheck] {name:<10} {getattr(m, '__version__', '?'):<14} {getattr(m, '__file__', '?')}")
    except Exception as e:
        ok = False
        print(f"[selfcheck] {name:<10} 导入失败: {e!r}")
try:
    importlib.import_module('projects.BEVFusion.bevfusion.ops.voxel.voxel_layer')
    importlib.import_module('projects.BEVFusion.bevfusion.ops.bev_pool.bev_pool_ext')
    print('[selfcheck] BEVFusion ops (voxel_layer / bev_pool_ext) 导入 OK')
except Exception as e:
    ok = False
    print(f"[selfcheck] BEVFusion ops 导入失败: {e!r}")
try:
    import torch
    print(f"[selfcheck] CUDA available={torch.cuda.is_available()} "
          f"device_count={torch.cuda.device_count()}")
except Exception:
    pass
sys.exit(0 if ok else 1)
PYEOF
RC=$?
set -e
if [[ ${RC} -ne 0 ]]; then
  echo "[deploy] 自检未全部通过（见上）" >&2
  exit ${RC}
fi

echo "[deploy] 完成。后续命令均从本目录出发，例如："
echo "  bash scripts/m0_probe.sh --config projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_4xa30-amp-accum_nus-3d.py \\"
echo "       --repro-ckpt work_dirs/baseline1/epoch_19.pth --official-ckpt <path>/bevfusion_lidar-cam_..._nus-3d-5239b1af.pth \\"
echo "       --work-dir work_dirs/<run>"
