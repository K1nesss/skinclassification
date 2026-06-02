#!/usr/bin/env bash
set -euo pipefail

# Create data/output symlinks for servers with small home disks.
# Usage:
#   bash scripts/setup_server_storage.sh [project_root] [storage_root]

PROJECT_ROOT="${1:-$(pwd)}"
STORAGE_ROOT="${2:-${SKIN_STORAGE_ROOT:-/mnt/disk002/skinclassification}}"

mkdir -p "$STORAGE_ROOT/data/raw"
mkdir -p "$STORAGE_ROOT/data/new"
mkdir -p "$STORAGE_ROOT/data/pass"
mkdir -p "$STORAGE_ROOT/data/interim"
mkdir -p "$STORAGE_ROOT/data/processed"
mkdir -p "$STORAGE_ROOT/outputs/checkpoints"
mkdir -p "$STORAGE_ROOT/outputs/logs"
mkdir -p "$STORAGE_ROOT/outputs/figures"
mkdir -p "$STORAGE_ROOT/outputs/gradcam"
mkdir -p "$STORAGE_ROOT/outputs/feature_maps"
mkdir -p "$STORAGE_ROOT/outputs/error_cases"
mkdir -p "$STORAGE_ROOT/outputs/reports"

link_dir() {
  local name="$1"
  local target="$STORAGE_ROOT/$name"
  local link="$PROJECT_ROOT/$name"

  if [[ -L "$link" ]]; then
    local current
    current="$(readlink "$link")"
    if [[ "$current" == "$target" ]]; then
      echo "$name 已经链接到 $target"
      return
    fi
    echo "更新 $name 软链接：$current -> $target"
    rm "$link"
  elif [[ -e "$link" ]]; then
    if [[ -d "$link" ]] && [[ -z "$(find "$link" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      rmdir "$link"
    elif [[ -d "$link" ]] && [[ "${MOVE_EXISTING:-0}" == "1" ]]; then
      echo "正在把已有 $link 内容移动到 $target"
      shopt -s dotglob nullglob
      mv "$link"/* "$target"/
      shopt -u dotglob nullglob
      rmdir "$link"
    else
      echo "拒绝替换非空目录：$link" >&2
      echo "请先把内容移动到 $target，或上传代码时不要包含本地 $name 目录。" >&2
      echo "也可以使用 MOVE_EXISTING=1 自动移动。" >&2
      exit 1
    fi
  fi

  ln -s "$target" "$link"
  echo "已创建软链接：$link -> $target"
}

link_dir data
link_dir outputs
