#!/usr/bin/env bash
# deploy.sh
# ---------
# Sets up the environment and launches QLoRA training for the adversary model.
# Training runs on whatever GPUs are present — no mode switching required.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REPO_ROOT}/venv"
PYTHON_BIN="python3"

echo "============================================================"
echo " Adversary LoRA Pipeline — Launch"
echo " $(date)"
echo "============================================================"

# ---------------------------------------------------------------------------
# 1. Create virtual environment
# ---------------------------------------------------------------------------
if [[ ! -d "${VENV_DIR}" ]]; then
    echo "[1/3] Creating virtual environment at ${VENV_DIR} …"
    ${PYTHON_BIN} -m venv "${VENV_DIR}"
else
    echo "[1/3] Using existing virtual environment at ${VENV_DIR}."
fi
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip --quiet

# ---------------------------------------------------------------------------
# 2. Install dependencies
# ---------------------------------------------------------------------------
echo "[2/3] Installing dependencies from requirements.txt …"
pip install -r "${REPO_ROOT}/requirements.txt" --quiet
echo "      Done."

# ---------------------------------------------------------------------------
# 3. Launch training
# ---------------------------------------------------------------------------
echo "[3/3] Launching training …"
accelerate launch \
    --config_file "${REPO_ROOT}/train/accelerate_config.yaml" \
    "${REPO_ROOT}/train/train_qlora.py" \
    --output-dir "${REPO_ROOT}/output/adversary-lora"

