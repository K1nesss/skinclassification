#!/usr/bin/env bash
set -euo pipefail

# Run after activating the conda environment.
# Example:
#   conda activate pytorch
#   bash run_all_experiments.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${CONFIG:-config.yaml}"
PYTHON="${PYTHON:-python}"
STORAGE_ROOT="${SKIN_STORAGE_ROOT:-/mnt/disk002/skinclassification}"

# Full experiment set. Override if needed:
#   MODELS="convnext_base swin_b" bash run_all_experiments.sh
MODELS_STRING="${MODELS:-resnet18 densenet121 efficientnet_b0 mobilenet_v3_small convnext_tiny convnext_base swin_s swin_b}"
read -r -a MODELS_ARRAY <<< "$MODELS_STRING"

RUN_BALANCE_ABLATION="${RUN_BALANCE_ABLATION:-1}"
ABLATION_MODEL="${ABLATION_MODEL:-convnext_base}"
RUN_FULL_ANALYSIS="${RUN_FULL_ANALYSIS:-1}"
ANALYSIS_MODEL="${ANALYSIS_MODEL:-}"

echo "================================================================================"
echo "完整实验总脚本启动"
echo "================================================================================"
echo "项目目录：$PROJECT_ROOT"
echo "配置文件：$CONFIG"
echo "Python 命令：$PYTHON"
echo "大盘存储目录：$STORAGE_ROOT"
echo "计划训练模型：${MODELS_ARRAY[*]}"
echo "类别平衡消融：$RUN_BALANCE_ABLATION，消融模型：$ABLATION_MODEL"
echo "完整分析产物：$RUN_FULL_ANALYSIS"
echo "提示：如需临时少跑几轮，可用 EPOCHS=5 bash run_all_experiments.sh"

echo
echo "检查并创建 data/outputs 软链接"
bash scripts/setup_server_storage.sh "$PROJECT_ROOT" "$STORAGE_ROOT"

echo
echo "[1/8] 构建新十分类数据集、质量清洗、划分 train/val/test"
if [[ "${SKIP_PREPARE:-0}" == "1" ]]; then
  echo "SKIP_PREPARE=1，跳过数据准备。"
else
  prepare_args=(scripts/build_new_dataset.py --config "$CONFIG")
  if [[ "${CLEAN_DATASET:-1}" == "1" ]]; then
    prepare_args+=(--clean)
  fi
  "$PYTHON" "${prepare_args[@]}"
fi

echo
echo "[2/8] 生成数据集、图像样本与增强可视化图表"
"$PYTHON" scripts/generate_all_figures.py --config "$CONFIG"

echo
echo "[3/8] 生成数据集统计报告"
"$PYTHON" scripts/make_dataset_report.py --config "$CONFIG"

batch_size_for_model() {
  case "$1" in
    convnext_base) echo "${BATCH_SIZE_CONVNEXT_BASE:-8}" ;;
    swin_b) echo "${BATCH_SIZE_SWIN_B:-8}" ;;
    swin_s) echo "${BATCH_SIZE_SWIN_S:-8}" ;;
    convnext_tiny) echo "${BATCH_SIZE_CONVNEXT_TINY:-16}" ;;
    mobilenet_v3_small) echo "${BATCH_SIZE_MOBILENET_V3_SMALL:-32}" ;;
    *) echo "${BATCH_SIZE:-16}" ;;
  esac
}

echo
echo "[4/8] 开始训练模型：${MODELS_ARRAY[*]}"
for model in "${MODELS_ARRAY[@]}"; do
  batch_size="$(batch_size_for_model "$model")"
  echo
  echo "--------------------------------------------------------------------------------"
  echo "训练模型：$model，batch size：$batch_size"
  echo "--------------------------------------------------------------------------------"
  train_args=(train.py --config "$CONFIG" --model "$model" --batch-size "$batch_size")
  if [[ -n "${EPOCHS:-}" ]]; then
    train_args+=(--epochs "$EPOCHS")
  fi
  "$PYTHON" "${train_args[@]}"
done

echo
echo "[5/8] 开始评估所有 checkpoint"
for model in "${MODELS_ARRAY[@]}"; do
  checkpoint="outputs/checkpoints/${model}_best.pt"
  batch_size="$(batch_size_for_model "$model")"
  if [[ ! -f "$checkpoint" ]]; then
    echo "缺少 checkpoint：$checkpoint" >&2
    exit 1
  fi
  echo
  echo "评估内部测试集：$checkpoint"
  "$PYTHON" evaluate.py --config "$CONFIG" --checkpoint "$checkpoint" --split test --batch-size "$batch_size"
done

echo
echo "[6/8] 类别平衡消融实验"
if [[ "$RUN_BALANCE_ABLATION" == "1" ]]; then
  ablation_batch_size="$(batch_size_for_model "$ABLATION_MODEL")"
  ablation_args=(
    scripts/run_balance_ablation.py
    --config "$CONFIG"
    --model "$ABLATION_MODEL"
    --batch-size "$ablation_batch_size"
  )
  if [[ -n "${ABLATION_EPOCHS:-${EPOCHS:-}}" ]]; then
    ablation_args+=(--epochs "${ABLATION_EPOCHS:-${EPOCHS:-}}")
  fi
  "$PYTHON" "${ablation_args[@]}"
else
  echo "RUN_BALANCE_ABLATION=0，跳过类别平衡消融实验。"
fi

echo
echo "[7/8] 生成完整分析图表：ROC/PR、校准、错误样本、Grad-CAM、特征图、t-SNE/UMAP、Demo 展示图"
if [[ "$RUN_FULL_ANALYSIS" == "1" ]]; then
  analysis_args=(scripts/generate_full_analysis.py --config "$CONFIG")
  if [[ -n "$ANALYSIS_MODEL" ]]; then
    analysis_args+=(--model "$ANALYSIS_MODEL")
  fi
  "$PYTHON" "${analysis_args[@]}"
else
  echo "RUN_FULL_ANALYSIS=0，跳过完整分析产物生成。"
fi

echo
echo "[8/8] 全部实验完成"
echo "报告目录：outputs/reports"
echo "图表目录：outputs/figures"
echo "Grad-CAM 目录：outputs/gradcam"
echo "特征图目录：outputs/feature_maps"
echo "错误样本目录：outputs/error_cases"
echo "模型目录：outputs/checkpoints"
