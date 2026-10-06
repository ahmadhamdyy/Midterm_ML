#!/usr/bin/env bash
#SBATCH --job-name=portrait-smoke-pvc
#SBATCH --partition=pvc
#SBATCH --gres=gpu:pvc:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/user/u.ae271721/midterm_mtl/logs/pvc-smoke-%j.out

set -euo pipefail
cd "$SCRATCH/midterm_mtl"
module load intelpython/2024.1.0_814 intel/2023.07
source /sw/hprc/sw/Python/virtualenvs/intelpython/2024.1.0_814/intel-ai-python-env/bin/activate
set +u
source /sw/hprc/sw/oneAPI/2024.2/setvars.sh
set -u
export USE_IPEX=1
exec python portrait_mtl.py smoke --device xpu
