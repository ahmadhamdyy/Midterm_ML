#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH:?SCRATCH is required}/midterm_mtl}"
cd "$PROJECT_ROOT"
mkdir -p artifacts logs runs

module purge
module load GCC/13.3.0 OpenMPI/5.0.3 PyTorch/2.9.1-CUDA-12.6.0
python -m venv --system-site-packages .venv-h100
.venv-h100/bin/python -m pip install --upgrade pip
.venv-h100/bin/python -m pip install filelock gdown jupyter matplotlib pandas scikit-learn seaborn reportlab pytest
.venv-h100/bin/python -m pip install --no-deps torchvision==0.24.1

# PVC jobs use ACES's supported shared Intel/IPEX environment (see the
# slurm/*_pvc.sbatch scripts), avoiding a large duplicate XPU installation.

echo "ACES environments are ready in $PROJECT_ROOT"
