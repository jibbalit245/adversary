#!/usr/bin/env bash
# deploy.sh
# ---------
# Deployment script for the adversary LoRA fine-tuning pipeline.
# Compatible with RunPod, Lambda Labs, and equivalent cloud GPU environments.
#
# Steps performed:
#   1. Create a Python 3.11 virtual environment
#   2. Install all dependencies from requirements.txt
#   3. Detect available GPUs (no hardcoded count)
#   4. Run a single dry-run step to verify no import or config errors
#   5. Run full training (single-GPU or multi-GPU depending on availability)
#
# Usage:
#   bash deploy.sh [--dry-run-only]
#
# Set DRY_RUN_ONLY=1 or pass --dry-run-only to stop after the verification step.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REPO_ROOT}/venv"
PYTHON_BIN="python3"
DRY_RUN_ONLY="${DRY_RUN_ONLY:-0}"

if [[ "${1:-}" == "--dry-run-only" ]]; then
    DRY_RUN_ONLY=1
fi

echo "============================================================"
echo " Adversary LoRA Pipeline — Deployment"
echo " $(date)"
echo "============================================================"

# ---------------------------------------------------------------------------
# 1. Python version check
# ---------------------------------------------------------------------------
PYTHON_VERSION=$(${PYTHON_BIN} --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo "${PYTHON_VERSION}" | cut -d. -f1)
PYTHON_MINOR=$(echo "${PYTHON_VERSION}" | cut -d. -f2)
echo "[1/5] Python ${PYTHON_VERSION} detected."
if [[ "${PYTHON_MAJOR}" -lt 3 || ( "${PYTHON_MAJOR}" -eq 3 && "${PYTHON_MINOR}" -lt 11 ) ]]; then
    echo "ERROR: Python 3.11+ is required (found ${PYTHON_VERSION})."
    echo "       Install it with: apt-get install python3.11"
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Create virtual environment
# ---------------------------------------------------------------------------
if [[ ! -d "${VENV_DIR}" ]]; then
    echo "[2/5] Creating virtual environment at ${VENV_DIR} …"
    ${PYTHON_BIN} -m venv "${VENV_DIR}"
else
    echo "[2/5] Virtual environment already exists at ${VENV_DIR}."
fi
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip --quiet

# ---------------------------------------------------------------------------
# 3. Install dependencies
# ---------------------------------------------------------------------------
echo "[3/5] Installing dependencies from requirements.txt …"
pip install -r "${REPO_ROOT}/requirements.txt" --quiet
echo "      Dependencies installed."

# ---------------------------------------------------------------------------
# 4. GPU detection
# ---------------------------------------------------------------------------
echo "[4/5] Detecting GPUs …"
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "      torch.cuda.device_count() = ${NUM_GPUS}"

if command -v nvidia-smi &>/dev/null; then
    echo "      GPU details:"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version \
        --format=csv,noheader | sed 's/^/        /'
fi

CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda or 'N/A')")
echo "      CUDA version (torch): ${CUDA_VERSION}"

if [[ "${NUM_GPUS}" -lt 1 ]]; then
    echo "WARNING: No CUDA GPUs detected. Training will be extremely slow on CPU."
fi

# ---------------------------------------------------------------------------
# 5. Dry-run verification
# ---------------------------------------------------------------------------
echo "[5/5] Running dry-run to verify pipeline (5 training steps) …"
python "${REPO_ROOT}/train/train_qlora.py" --dry-run
echo "      Dry-run passed ✓"

if [[ "${DRY_RUN_ONLY}" -eq 1 ]]; then
    echo ""
    echo "DRY_RUN_ONLY=1 — stopping after verification."
    echo "To launch full training, re-run without --dry-run-only."
    exit 0
fi

# ---------------------------------------------------------------------------
# Full training launch
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " Launching full training …"
echo "============================================================"

if [[ "${NUM_GPUS}" -le 1 ]]; then
    echo "Single-GPU mode."
    bash "${REPO_ROOT}/scripts/launch_single_gpu.sh" --full
else
    echo "Multi-GPU mode (${NUM_GPUS} GPUs)."
    bash "${REPO_ROOT}/scripts/launch_multi_gpu.sh"
fi
