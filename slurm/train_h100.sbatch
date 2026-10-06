#!/usr/bin/env bash
#SBATCH --job-name=portrait-mtl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/user/u.ae271721/midterm_mtl/logs/%x-%j.out

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$SCRATCH/midterm_mtl}"
cd "$PROJECT_ROOT"
module purge
module load GCC/13.3.0 OpenMPI/5.0.3 PyTorch/2.9.1-CUDA-12.6.0

exec .venv-h100/bin/python portrait_mtl.py train \
  --data-root "$PROJECT_ROOT/data" \
  --output-dir "$PROJECT_ROOT/runs/$RUN_NAME" \
  --task "$TASK" \
  --auxiliary-weight "${AUX_WEIGHT:-0.1}" \
  --epochs "${EPOCHS:-20}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --workers "${WORKERS:-12}" \
  --device cuda \
  --resume
