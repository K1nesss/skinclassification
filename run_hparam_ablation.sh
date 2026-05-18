#!/usr/bin/env bash
set -euo pipefail

# Run after activating the conda environment.
# Example:
#   conda activate pytorch
#   bash run_hparam_ablation.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${CONFIG:-config.yaml}"
PYTHON="${PYTHON:-python}"
MODEL="${HPARAM_MODEL:-convnext_base}"
EPOCHS_VALUE="${HPARAM_EPOCHS:-12}"
BATCH_SIZE_VALUE="${HPARAM_BATCH_SIZE:-16}"
DEVICE_VALUE="${HPARAM_DEVICE:-}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
INCLUDE_IMAGE_SIZE="${INCLUDE_IMAGE_SIZE:-1}"

echo "================================================================================"
echo "超参数消融实验脚本启动"
echo "================================================================================"
echo "项目目录：$PROJECT_ROOT"
echo "配置文件：$CONFIG"
echo "Python 命令：$PYTHON"
echo "主力模型：$MODEL"
echo "训练轮数：$EPOCHS_VALUE"
echo "batch size：$BATCH_SIZE_VALUE"
echo "是否包含 image_size 实验：$INCLUDE_IMAGE_SIZE"
echo "是否跳过已有结果：$SKIP_EXISTING"

args=(
  scripts/run_hparam_ablation.py
  --config "$CONFIG"
  --model "$MODEL"
  --epochs "$EPOCHS_VALUE"
  --batch-size "$BATCH_SIZE_VALUE"
)

if [[ -n "$DEVICE_VALUE" ]]; then
  args+=(--device "$DEVICE_VALUE")
fi

if [[ "$SKIP_EXISTING" == "1" ]]; then
  args+=(--skip-existing)
fi

if [[ "$INCLUDE_IMAGE_SIZE" == "1" ]]; then
  args+=(--include-image-size)
else
  args+=(--no-image-size)
fi

echo
echo "执行超参数实验："
echo "$PYTHON ${args[*]}"
echo
"$PYTHON" "${args[@]}"

echo
echo "================================================================================"
echo "超参数消融实验完成"
echo "================================================================================"
echo "报告输出：outputs/reports/${MODEL}_hparam_ablation_summary.md"
echo "CSV 输出：outputs/reports/${MODEL}_hparam_ablation_results.csv"
echo "图表输出："
echo "  outputs/figures/48_hparam_image_size_ablation.png"
echo "  outputs/figures/49_hparam_learning_rate_ablation.png"
echo "  outputs/figures/50_hparam_weight_decay_ablation.png"
echo "  outputs/figures/51_hparam_augmentation_ablation.png"
