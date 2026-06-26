#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/home/jmuznkx/miniconda3/envs/sjba/bin/python}"
GPU_INDEX="${GPU_INDEX:-0}"
IMAGE_SIZE="${IMAGE_SIZE:-384}"
FULL_MANIFEST="${FULL_MANIFEST:-data/interim/split_samples_manual_new1.csv}"
STRICT_MANIFEST="${STRICT_MANIFEST:-data/interim/split_samples_manual_new1_strict_no_scin_ed.csv}"
LOG_ROOT="${LOG_ROOT:-outputs/logs/manual_new1_training}"
mkdir -p "$LOG_ROOT" outputs/reports outputs/checkpoints

wait_for_gpu0() {
  local max_mem="${1:-4096}"
  local max_util="${2:-20}"
  while true; do
    local line mem util
    line="$(nvidia-smi --id="$GPU_INDEX" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"
    mem="$(echo "$line" | awk -F, '{gsub(/ /, "", $1); print $1}')"
    util="$(echo "$line" | awk -F, '{gsub(/ /, "", $2); print $2}')"
    if [[ "$mem" -le "$max_mem" && "$util" -le "$max_util" ]]; then
      echo "GPU${GPU_INDEX} ready: memory=${mem}MiB util=${util}%"
      return
    fi
    echo "Waiting for GPU${GPU_INDEX}: memory=${mem}MiB util=${util}%"
    sleep 300
  done
}

run_train() {
  local run_name="$1"
  shift
  local ckpt="outputs/checkpoints/${run_name}_best.pt"
  local log_file="${LOG_ROOT}/${run_name}.log"
  if [[ -f "$ckpt" ]]; then
    echo "Skip existing checkpoint: $ckpt"
    return
  fi
  echo "Start training: $run_name"
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$PYTHON_BIN" train.py "$@" --run-name "$run_name" 2>&1 | tee "$log_file"
}

run_eval() {
  local run_name="$1"
  local manifest="$2"
  local batch_size="$3"
  local ckpt="outputs/checkpoints/${run_name}_best.pt"
  if [[ ! -f "$ckpt" ]]; then
    echo "Missing checkpoint for eval: $ckpt" >&2
    return 1
  fi
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$PYTHON_BIN" evaluate.py \
    --config config.yaml \
    --manifest "$manifest" \
    --checkpoint "$ckpt" \
    --split test \
    --batch-size "$batch_size" \
    --image-size "$IMAGE_SIZE" \
    2>&1 | tee "${LOG_ROOT}/${run_name}_test_eval.log"
}

COMMON_ARGS=(
  --config config.yaml
  --image-size "$IMAGE_SIZE"
  --epochs 30
  --early-stopping-patience 30
  --loss-type focal
  --focal-gamma 1.5
  --label-smoothing 0.05
  --balance-strategy class_aware_sampler
  --hard-class-multiplier 1.2
  --no-loss-class-weights
)

CONVNEXT_FULL="convnext_base_manual_new1_full_b32_gpu0"
SWIN_FULL="swin_b_manual_new1_full_b8acc4_gpu0"
CONVNEXT_STRICT="convnext_base_manual_new1_strict_b32_gpu0"
CONVNEXT_STRICT_TARGET="convnext_base_manual_new1_strict_targetboost_b32_gpu0"
SWIN_STRICT="swin_b_manual_new1_strict_b8acc4_gpu0"

echo "Manual new_1 five-model training started at $(date -Is)"
echo "Full manifest: $FULL_MANIFEST"
echo "Strict manifest: $STRICT_MANIFEST"

wait_for_gpu0 4096 20
run_train "$CONVNEXT_FULL" \
  "${COMMON_ARGS[@]}" \
  --manifest "$FULL_MANIFEST" \
  --model convnext_base \
  --batch-size 32 &
pid1=$!
run_train "$SWIN_FULL" \
  "${COMMON_ARGS[@]}" \
  --manifest "$FULL_MANIFEST" \
  --model swin_b \
  --batch-size 8 \
  --grad-accum-steps 4 &
pid2=$!
wait "$pid1" "$pid2"

wait_for_gpu0 4096 20
run_train "$CONVNEXT_STRICT_TARGET" \
  --config config.yaml \
  --manifest "$STRICT_MANIFEST" \
  --model convnext_base \
  --image-size "$IMAGE_SIZE" \
  --batch-size 32 \
  --epochs 30 \
  --early-stopping-patience 30 \
  --loss-type focal \
  --focal-gamma 1.5 \
  --label-smoothing 0.03 \
  --balance-strategy class_aware_sampler \
  --hard-class-multiplier 2.0 \
  --no-loss-class-weights &
pid1=$!
run_train "$SWIN_STRICT" \
  "${COMMON_ARGS[@]}" \
  --manifest "$STRICT_MANIFEST" \
  --model swin_b \
  --batch-size 8 \
  --grad-accum-steps 4 &
pid2=$!
wait "$pid1" "$pid2"

wait_for_gpu0 4096 20
run_train "$CONVNEXT_STRICT" \
  "${COMMON_ARGS[@]}" \
  --manifest "$STRICT_MANIFEST" \
  --model convnext_base \
  --batch-size 32

echo "Training finished at $(date -Is)"
echo "Start individual test evaluation"
run_eval "$CONVNEXT_FULL" "$FULL_MANIFEST" 32
run_eval "$SWIN_FULL" "$FULL_MANIFEST" 32
run_eval "$CONVNEXT_STRICT" "$STRICT_MANIFEST" 32
run_eval "$CONVNEXT_STRICT_TARGET" "$STRICT_MANIFEST" 32
run_eval "$SWIN_STRICT" "$STRICT_MANIFEST" 32

echo "Start five-model ensemble evaluation"
CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$PYTHON_BIN" scripts/evaluate_ensemble.py \
  --config config.yaml \
  --manifest "$STRICT_MANIFEST" \
  --image-size "$IMAGE_SIZE" \
  --batch-size 32 \
  --split test \
  --search-on-val \
  --weight-step 0.05 \
  --output-name ensemble_manual_new1_full_plus_strict5_test_metrics \
  --checkpoints \
    "outputs/checkpoints/${CONVNEXT_FULL}_best.pt" \
    "outputs/checkpoints/${SWIN_FULL}_best.pt" \
    "outputs/checkpoints/${CONVNEXT_STRICT}_best.pt" \
    "outputs/checkpoints/${CONVNEXT_STRICT_TARGET}_best.pt" \
    "outputs/checkpoints/${SWIN_STRICT}_best.pt" \
  2>&1 | tee "${LOG_ROOT}/ensemble_manual_new1_full_plus_strict5_test_metrics.log"

echo "Manual new_1 five-model training and evaluation finished at $(date -Is)"
