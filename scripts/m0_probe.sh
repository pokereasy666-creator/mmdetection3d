#!/usr/bin/env bash
# m0_probe.sh — m0_probe.py 的薄包装：把仓库根目录放进 PYTHONPATH 后转发全部参数。
# 保证 mmdet3d 与 projects.* 从【当前解压目录】解析（而非旧 editable 安装），
# 与 tools/dist_train.sh 的 PYTHONPATH 约定一致。
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
exec "${PYTHON:-python}" "${REPO_ROOT}/scripts/m0_probe.py" "$@"
