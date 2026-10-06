#!/usr/bin/env bash
#SBATCH --job-name=portrait-final
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/user/u.ae271721/midterm_mtl/logs/%x-%j.out

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$SCRATCH/midterm_mtl}"
cd "$PROJECT_ROOT"
module purge
module load GCC/13.3.0 OpenMPI/5.0.3 PyTorch/2.9.1-CUDA-12.6.0

readarray -t CHECKPOINTS < <(.venv-h100/bin/python - <<'PY'
import json
selection = json.load(open("artifacts/model_selection.json"))
print(selection["single_checkpoint"])
print(selection["multi_checkpoint"])
PY
)

exec .venv-h100/bin/python portrait_mtl.py evaluate \
  --data-root "$PROJECT_ROOT/data" \
  --single-checkpoint "${CHECKPOINTS[0]}" \
  --multi-checkpoint "${CHECKPOINTS[1]}" \
  --output-dir "$PROJECT_ROOT/artifacts/final" \
  --workers 12 \
  --device cuda
