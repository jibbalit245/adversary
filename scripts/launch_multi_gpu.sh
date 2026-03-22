#!/usr/bin/env bash
# launch_multi_gpu.sh
# -------------------
# Multi-GPU production launch for the adversary QLoRA training pipeline.
# Uses `accelerate launch` to auto-detect and use ALL available GPUs.
# GPU count is NOT hardcoded — detected at runtime via torch.cuda.device_count().
#
# Usage:
#   bash scripts/launch_multi_gpu.sh
#
# To restrict to specific GPUs:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/launch_multi_gpu.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_SCRIPT="${REPO_ROOT}/train/train_qlora.py"
ACCEL_CONFIG="${REPO_ROOT}/train/accelerate_config.yaml"

# Detect number of available GPUs at runtime
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "[multi-GPU] Detected ${NUM_GPUS} GPU(s)."

if [[ "${NUM_GPUS}" -lt 1 ]]; then
    echo "ERROR: No GPUs detected. Aborting."
    exit 1
fi

if [[ "${NUM_GPUS}" -eq 1 ]]; then
    echo "[multi-GPU] Only 1 GPU available — falling back to single-GPU launch."
    exec bash "$(dirname "${BASH_SOURCE[0]}")/launch_single_gpu.sh" "$@"
fi

echo "[multi-GPU] Launching on ${NUM_GPUS} GPU(s) via accelerate."
echo "GPU details:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null \
    || echo "  (nvidia-smi not available)"

accelerate launch \
    --config_file "${ACCEL_CONFIG}" \
    --num_processes "${NUM_GPUS}" \
    "${TRAIN_SCRIPT}" \
    --output-dir "${REPO_ROOT}/output/adversary-lora"
