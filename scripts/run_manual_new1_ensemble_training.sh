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

checkpoint_path() {
  local run_name="$1"
  echo "outputs/checkpoints/${run_name}_best.pt"
}

existing_run_name() {
  local run_name
  for run_name in "$@"; do
    if [[ -f "$(checkpoint_path "$run_name")" ]]; then
      SELECTED_RUN="$run_name"
      return 0
    fi
  done
  return 1
}

run_train_once() {
  local run_name="$1"
  shift
  local ckpt log_file
  ckpt="$(checkpoint_path "$run_name")"
  log_file="${LOG_ROOT}/${run_name}.log"
  if [[ -f "$ckpt" ]]; then
    echo "Skip existing checkpoint: $ckpt"
    return 0
  fi

  echo "Start training: $run_name"
  set +e
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$PYTHON_BIN" train.py "$@" --run-name "$run_name" 2>&1 | tee "$log_file"
  local status=${PIPESTATUS[0]}
  if [[ "$status" -eq 0 ]]; then
    return 0
  fi
  if grep -qE "OutOfMemory|CUDA out of memory" "$log_file"; then
    echo "OOM while training $run_name"
    return 99
  fi
  return "$status"
}

train_convnext_fallback() {
  local base_name="$1"
  local manifest="$2"
  shift 2
  local status
  if existing_run_name "${base_name}_b32_gpu0" "${base_name}_b16_gpu0" "${base_name}_b8_gpu0"; then
    echo "Use existing checkpoint: $(checkpoint_path "$SELECTED_RUN")"
    return 0
  fi

  while true; do
    set +e
    run_train_once "${base_name}_b32_gpu0" "$@" --manifest "$manifest" --model convnext_base --batch-size 32
    status=$?
    set -e
    if [[ "$status" -eq 0 ]]; then
      SELECTED_RUN="${base_name}_b32_gpu0"
      return 0
    fi
    if [[ "$status" -ne 99 ]]; then
      return 1
    fi

    set +e
    run_train_once "${base_name}_b16_gpu0" "$@" --manifest "$manifest" --model convnext_base --batch-size 16
    status=$?
    set -e
    if [[ "$status" -eq 0 ]]; then
      SELECTED_RUN="${base_name}_b16_gpu0"
      return 0
    fi
    if [[ "$status" -ne 99 ]]; then
      return 1
    fi

    set +e
    run_train_once "${base_name}_b8_gpu0" "$@" --manifest "$manifest" --model convnext_base --batch-size 8
    status=$?
    set -e
    if [[ "$status" -eq 0 ]]; then
      SELECTED_RUN="${base_name}_b8_gpu0"
      return 0
    fi
    if [[ "$status" -ne 99 ]]; then
      return 1
    fi

    echo "All ConvNeXt fallback batches OOM for ${base_name}; waiting before retry."
    wait_for_gpu0 4096 20
  done
}

train_swin_b8acc4_wait() {
  local run_name="$1"
  local manifest="$2"
  shift 2
  local status
  if [[ -f "$(checkpoint_path "$run_name")" ]]; then
    echo "Use existing checkpoint: $(checkpoint_path "$run_name")"
    SELECTED_RUN="$run_name"
    return 0
  fi

  while true; do
    set +e
    run_train_once "$run_name" "$@" --manifest "$manifest" --model swin_b --batch-size 8 --grad-accum-steps 4
    status=$?
    set -e
    if [[ "$status" -eq 0 ]]; then
      SELECTED_RUN="$run_name"
      return 0
    fi
    if [[ "$status" -ne 99 ]]; then
      return 1
    fi
    echo "Swin-B b8acc4 OOM for ${run_name}; waiting before retry."
    wait_for_gpu0 4096 20
  done
}

run_eval() {
  local run_name="$1"
  local manifest="$2"
  local ckpt
  ckpt="$(checkpoint_path "$run_name")"
  if [[ ! -f "$ckpt" ]]; then
    echo "Missing checkpoint for eval: $ckpt" >&2
    return 1
  fi
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$PYTHON_BIN" evaluate.py \
    --config config.yaml \
    --manifest "$manifest" \
    --checkpoint "$ckpt" \
    --split test \
    --batch-size 32 \
    --image-size "$IMAGE_SIZE" \
    2>&1 | tee "${LOG_ROOT}/${run_name}_test_eval.log"
}

echo "Manual new_1 five-model training started at $(date -Is)"
echo "Full manifest: $FULL_MANIFEST"
echo "Strict manifest: $STRICT_MANIFEST"

train_convnext_fallback convnext_base_manual_new1_full "$FULL_MANIFEST" "${COMMON_ARGS[@]}"
CONVNEXT_FULL="$SELECTED_RUN"
train_swin_b8acc4_wait swin_b_manual_new1_full_b8acc4_gpu0 "$FULL_MANIFEST" "${COMMON_ARGS[@]}"
SWIN_FULL="$SELECTED_RUN"
train_convnext_fallback \
  convnext_base_manual_new1_strict_targetboost \
  "$STRICT_MANIFEST" \
  --config config.yaml \
  --image-size "$IMAGE_SIZE" \
  --epochs 30 \
  --early-stopping-patience 30 \
  --loss-type focal \
  --focal-gamma 1.5 \
  --label-smoothing 0.03 \
  --balance-strategy class_aware_sampler \
  --hard-class-multiplier 2.0 \
  --no-loss-class-weights
CONVNEXT_STRICT_TARGET="$SELECTED_RUN"
train_swin_b8acc4_wait swin_b_manual_new1_strict_b8acc4_gpu0 "$STRICT_MANIFEST" "${COMMON_ARGS[@]}"
SWIN_STRICT="$SELECTED_RUN"
train_convnext_fallback convnext_base_manual_new1_strict "$STRICT_MANIFEST" "${COMMON_ARGS[@]}"
CONVNEXT_STRICT="$SELECTED_RUN"

echo "Selected runs:"
echo "  CONVNEXT_FULL=$CONVNEXT_FULL"
echo "  SWIN_FULL=$SWIN_FULL"
echo "  CONVNEXT_STRICT=$CONVNEXT_STRICT"
echo "  CONVNEXT_STRICT_TARGET=$CONVNEXT_STRICT_TARGET"
echo "  SWIN_STRICT=$SWIN_STRICT"

echo "Training finished at $(date -Is)"
echo "Start individual test evaluation"
run_eval "$CONVNEXT_FULL" "$FULL_MANIFEST"
run_eval "$SWIN_FULL" "$FULL_MANIFEST"
run_eval "$CONVNEXT_STRICT" "$STRICT_MANIFEST"
run_eval "$CONVNEXT_STRICT_TARGET" "$STRICT_MANIFEST"
run_eval "$SWIN_STRICT" "$STRICT_MANIFEST"

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
    "$(checkpoint_path "$CONVNEXT_FULL")" \
    "$(checkpoint_path "$SWIN_FULL")" \
    "$(checkpoint_path "$CONVNEXT_STRICT")" \
    "$(checkpoint_path "$CONVNEXT_STRICT_TARGET")" \
    "$(checkpoint_path "$SWIN_STRICT")" \
  2>&1 | tee "${LOG_ROOT}/ensemble_manual_new1_full_plus_strict5_test_metrics.log"

echo "Manual new_1 five-model training and evaluation finished at $(date -Is)"
