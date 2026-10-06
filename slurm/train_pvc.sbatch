#!/usr/bin/env bash
#SBATCH --job-name=portrait-mtl-pvc
#SBATCH --partition=pvc
#SBATCH --gres=gpu:pvc:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/user/u.ae271721/midterm_mtl/logs/%x-%j.out

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$SCRATCH/midterm_mtl}"
cd "$PROJECT_ROOT"
module load intelpython/2024.1.0_814 intel/2023.07
source /sw/hprc/sw/Python/virtualenvs/intelpython/2024.1.0_814/intel-ai-python-env/bin/activate
set +u
source /sw/hprc/sw/oneAPI/2024.2/setvars.sh
set -u
export USE_IPEX=1

exec python portrait_mtl.py train \
  --data-root "$PROJECT_ROOT/data" \
  --output-dir "$PROJECT_ROOT/runs/$RUN_NAME" \
  --task "$TASK" \
  --auxiliary-weight "${AUX_WEIGHT:-0.1}" \
  --epochs "${EPOCHS:-20}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --workers "${WORKERS:-12}" \
  --device xpu \
  --resume
