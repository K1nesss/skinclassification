#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/home/jmuznkx/miniconda3/envs/sjba/bin/python}"
GPU_INDEX="${GPU_INDEX:-0}"
IMAGE_SIZE="${IMAGE_SIZE:-384}"
MANIFEST="${MANIFEST:-data/interim/split_samples_manual_new1_strict_no_scin_ed.csv}"
MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME, for example timm:convnextv2_base.fcmae_ft_in22k_in1k_384}"
RUN_NAME="${RUN_NAME:?Set RUN_NAME, for example convnextv2_base_manual_new1_strict_b8_gpu0}"
LOG_ROOT="${LOG_ROOT:-outputs/logs/manual_new1_timm_trials}"
BATCHES="${BATCHES:-32 16 8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

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

run_train_once() {
  local batch_size="$1"
  local run_name="${RUN_NAME}_b${batch_size}_gpu0"
  local ckpt="outputs/checkpoints/${run_name}_best.pt"
  local log_file="${LOG_ROOT}/${run_name}.log"
  if [[ -f "$ckpt" ]]; then
    echo "Skip existing checkpoint: $ckpt"
    SELECTED_RUN="$run_name"
    return 0
  fi

  echo "Start training: $run_name"
  set +e
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$PYTHON_BIN" train.py \
    --config config.yaml \
    --manifest "$MANIFEST" \
    --model "$MODEL_NAME" \
    --run-name "$run_name" \
    --image-size "$IMAGE_SIZE" \
    --epochs 30 \
    --early-stopping-patience 30 \
    --batch-size "$batch_size" \
    --grad-accum-steps "$GRAD_ACCUM_STEPS" \
    --loss-type focal \
    --focal-gamma 1.5 \
    --label-smoothing 0.05 \
    --balance-strategy class_aware_sampler \
    --hard-class-multiplier 1.2 \
    --no-loss-class-weights \
    $EXTRA_TRAIN_ARGS \
    2>&1 | tee "$log_file"
  local status=${PIPESTATUS[0]}
  if [[ "$status" -eq 0 ]]; then
    SELECTED_RUN="$run_name"
    return 0
  fi
  if grep -qE "OutOfMemory|CUDA out of memory" "$log_file"; then
    echo "OOM while training $run_name"
    return 99
  fi
  return "$status"
}

run_eval_once() {
  local run_name="$1"
  local batch_size="$2"
  local ckpt="outputs/checkpoints/${run_name}_best.pt"
  local log_file="${LOG_ROOT}/${run_name}_test_eval_b${batch_size}.log"
  set +e
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$PYTHON_BIN" evaluate.py \
    --config config.yaml \
    --manifest "$MANIFEST" \
    --checkpoint "$ckpt" \
    --split test \
    --batch-size "$batch_size" \
    --image-size "$IMAGE_SIZE" \
    2>&1 | tee "$log_file"
  local status=${PIPESTATUS[0]}
  if [[ "$status" -eq 0 ]]; then
    return 0
  fi
  if grep -qE "OutOfMemory|CUDA out of memory" "$log_file"; then
    echo "OOM while evaluating $run_name with batch_size=$batch_size"
    return 99
  fi
  return "$status"
}

run_eval() {
  local run_name="$1"
  local batch_size status
  for batch_size in 32 16 8; do
    set +e
    run_eval_once "$run_name" "$batch_size"
    status=$?
    set -e
    if [[ "$status" -eq 0 ]]; then
      return 0
    fi
    if [[ "$status" -ne 99 ]]; then
      return 1
    fi
  done
  echo "All eval fallback batches OOM for ${run_name}; waiting before retry."
  wait_for_gpu0 4096 20
  run_eval "$run_name"
}

echo "Timm candidate training started at $(date -Is)"
echo "Model: $MODEL_NAME"
echo "Manifest: $MANIFEST"
echo "Image size: $IMAGE_SIZE"
echo "Batches: $BATCHES"

while true; do
  for batch_size in $BATCHES; do
    set +e
    run_train_once "$batch_size"
    status=$?
    set -e
    if [[ "$status" -eq 0 ]]; then
      echo "Selected run: $SELECTED_RUN"
      run_eval "$SELECTED_RUN"
      echo "Timm candidate finished at $(date -Is)"
      exit 0
    fi
    if [[ "$status" -ne 99 ]]; then
      exit "$status"
    fi
  done
  echo "All train fallback batches OOM for ${RUN_NAME}; waiting before retry."
  wait_for_gpu0 4096 20
done
