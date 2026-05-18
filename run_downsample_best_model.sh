#!/usr/bin/env bash
set -euo pipefail

# Run after activating the conda environment.
# Example:
#   conda activate pytorch
#   bash run_downsample_best_model.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${CONFIG:-config.yaml}"
PYTHON="${PYTHON:-python}"
STORAGE_ROOT="${SKIN_STORAGE_ROOT:-/mnt/disk002/skinclassification}"
MODEL="${DOWNSAMPLE_MODEL:-swin_b}"
EPOCHS_VALUE="${DOWNSAMPLE_EPOCHS:-30}"
BATCH_SIZE_VALUE="${DOWNSAMPLE_BATCH_SIZE:-16}"
MAX_OTHERS_PER_SOURCE="${MAX_OTHERS_PER_SOURCE:-3000}"
OTHERS_RATIO="${OTHERS_RATIO:-2.0}"

echo "================================================================================"
echo "最佳模型 train-only others 下采样三组对比实验"
echo "================================================================================"
echo "项目目录：$PROJECT_ROOT"
echo "配置文件：$CONFIG"
echo "Python 命令：$PYTHON"
echo "大盘存储目录：$STORAGE_ROOT"
echo "模型：$MODEL"
echo "训练轮数：$EPOCHS_VALUE"
echo "batch size：$BATCH_SIZE_VALUE"
echo "数据来源：Dermnet + Mendeley + SkinDisNet，暂不使用 SCIN"
echo "对比方案："
echo "  1. no_downsample：不下采样 + WeightedRandomSampler"
echo "  2. train_downsample_others${MAX_OTHERS_PER_SOURCE}：只对 train 的 others 每个数据源最多保留 ${MAX_OTHERS_PER_SOURCE} 张 + WeightedRandomSampler"
echo "  3. train_downsample_others_ratio2x：只对 train 的 others 控制为最大非 others 类数量的 ${OTHERS_RATIO} 倍，不使用 WeightedRandomSampler"

echo
echo "检查并创建 data/outputs 软链接"
bash scripts/setup_server_storage.sh "$PROJECT_ROOT" "$STORAGE_ROOT"

make_config() {
  local output_config="$1"
  local strategy="$2"
  "$PYTHON" - "$CONFIG" "$output_config" "$strategy" "$MAX_OTHERS_PER_SOURCE" "$OTHERS_RATIO" "$MODEL" "$EPOCHS_VALUE" "$BATCH_SIZE_VALUE" <<'PY'
import sys
from pathlib import Path
import yaml

src, dst, strategy, max_others, ratio, model, epochs, batch_size = sys.argv[1:9]
max_others = int(max_others)
ratio = float(ratio)
epochs = int(epochs)
batch_size = int(batch_size)

with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg["data"]["use_sources"] = ["dermnet", "skindisnet", "mendeley"]
cfg["data"]["split_external_sources"] = []
cfg["data"]["max_samples_per_class"] = None
cfg["data"]["downsample_majority"] = {
    "enabled": strategy != "none",
    "label": "others",
    "split": "train",
    "strategy": "max_per_source" if strategy == "max_per_source" else "ratio_to_max_non_majority",
    "max_per_source": max_others,
    "ratio_to_max_non_majority": ratio,
    "skip_external_sources": True,
}
cfg["training"]["model"] = model
cfg["training"]["epochs"] = epochs
cfg["training"]["batch_size"] = batch_size
cfg["training"]["balance_strategy"] = "weighted_sampler"
cfg["training"]["loss_class_weights"] = False

Path(dst).parent.mkdir(parents=True, exist_ok=True)
with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
PY
}

run_variant() {
  local variant="$1"
  local strategy="$2"
  local balance_strategy="$3"
  local temp_config="outputs/reports/${MODEL}_${variant}_config.yaml"
  local run_name="${MODEL}_${variant}"
  local checkpoint="outputs/checkpoints/${run_name}_best.pt"

  echo
  echo "--------------------------------------------------------------------------------"
  echo "开始实验：$variant，balance_strategy=$balance_strategy"
  echo "--------------------------------------------------------------------------------"
  make_config "$temp_config" "$strategy"

  echo
  echo "[A] 重新生成固定数据划分，并按方案只处理 train：$variant"
  "$PYTHON" scripts/prepare_dataset.py --config "$temp_config"

  echo
  echo "[B] 生成数据统计图和数据报告：$variant"
  "$PYTHON" scripts/generate_all_figures.py --config "$temp_config"
  "$PYTHON" scripts/make_dataset_report.py --config "$temp_config"
  cp outputs/reports/dataset_report.md "outputs/reports/${run_name}_dataset_report.md"

  echo
  echo "[C] 训练模型：$run_name"
  "$PYTHON" train.py \
    --config "$temp_config" \
    --model "$MODEL" \
    --run-name "$run_name" \
    --epochs "$EPOCHS_VALUE" \
    --batch-size "$BATCH_SIZE_VALUE" \
    --balance-strategy "$balance_strategy" \
    --no-loss-class-weights

  echo
  echo "[D] 评估内部 test：$run_name"
  "$PYTHON" evaluate.py \
    --config "$temp_config" \
    --checkpoint "$checkpoint" \
    --split test \
    --batch-size "$BATCH_SIZE_VALUE"
}

run_variant "no_downsample" "none" "weighted_sampler"
run_variant "train_downsample_others${MAX_OTHERS_PER_SOURCE}" "max_per_source" "weighted_sampler"
run_variant "train_downsample_others_ratio2x" "ratio_to_max_non_majority" "none"

echo
echo "[E] 生成三组对比汇总"
"$PYTHON" scripts/summarize_downsample_experiment.py \
  --model "$MODEL" \
  --max-others "$MAX_OTHERS_PER_SOURCE" \
  --ratio "$OTHERS_RATIO"

echo
echo "================================================================================"
echo "train-only others 下采样三组对比实验完成"
echo "================================================================================"
echo "汇总报告：outputs/reports/${MODEL}_train_downsample_comparison_summary.md"
echo "对比图：outputs/figures/${MODEL}_train_downsample_comparison.png"
