#!/usr/bin/env bash
# launch_single_gpu.sh
# --------------------
# Single-GPU debug launch for the adversary QLoRA training pipeline.
# Uses CUDA_VISIBLE_DEVICES=0 to pin to the first GPU.
#
# Usage:
#   bash scripts/launch_single_gpu.sh [--dry-run]
#
# Pass --dry-run to run only 5 training steps (no model download).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_SCRIPT="${REPO_ROOT}/train/train_qlora.py"

# --dry-run  → 5 steps only (default when no argument given)
# --full     → full training
# (no args)  → dry-run by default for safety
DRY_RUN_FLAG="--dry-run"
MODE="DRY-RUN"
if [[ "${1:-}" == "--full" ]]; then
    DRY_RUN_FLAG=""
    MODE="FULL"
elif [[ "${1:-}" == "--dry-run" || $# -eq 0 ]]; then
    DRY_RUN_FLAG="--dry-run"
    MODE="DRY-RUN"
fi
echo "[single-GPU] Launching in ${MODE} mode on GPU 0."

export CUDA_VISIBLE_DEVICES=0

echo "GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
    || echo "  (nvidia-smi not available)"

python "${TRAIN_SCRIPT}" \
    --output-dir "${REPO_ROOT}/output/adversary-lora" \
    ${DRY_RUN_FLAG}
